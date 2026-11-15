// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The ABX session: trials read from and reported to the terminal.

use std::collections::hash_map::RandomState;
use std::hash::BuildHasher;
use std::io::{BufRead, Write};

use crate::clip::Clip;
use crate::player::{Player, sound};

const HELP: &str = "  a, b, x  play A (the original), B (the encoding) or X (one of them)
  d        play the encoding less the original (the same in every trial)
  s        stop playing
  xa, xb   answer that X is A, or that X is B
  q        end the test early";

pub fn run(clip: &Clip, trials: usize) -> Result<(), String> {
    let a = sound(&clip.original);
    let b = sound(&clip.encoded);
    let d = sound(&clip.difference());
    let mut player = Player::new(clip.sample_rate, clip.channels);
    let seconds = clip.original.len() as f64 / (clip.channels as f64 * f64::from(clip.sample_rate));
    println!("{trials} trials of a {seconds:.1} s clip. Commands, each followed by Enter:\n{HELP}");

    let mut lines = std::io::stdin().lock().lines();
    let mut correct = 0;
    let mut taken = 0;
    'trials: for trial in 1..=trials {
        let x_is_b = coin();
        let x = if x_is_b { &b } else { &a };
        loop {
            print!("trial {trial}/{trials}> ");
            std::io::stdout().flush().map_err(|e| e.to_string())?;
            let Some(line) = lines.next() else {
                println!();
                break 'trials;
            };
            let line = line.map_err(|e| format!("reading stdin: {e}"))?;
            match line.trim() {
                "a" => player.play(&a)?,
                "b" => player.play(&b)?,
                "x" => player.play(x)?,
                "d" => player.play(&d)?,
                "s" => player.stop(),
                answer @ ("xa" | "xb") => {
                    player.stop();
                    taken += 1;
                    if (answer == "xb") == x_is_b {
                        correct += 1;
                        println!("  right: {correct} of {taken}");
                    } else {
                        println!("  wrong: {correct} of {taken}");
                    }
                    break;
                }
                "q" => break 'trials,
                "" => {}
                other => println!("  unknown command: {other}\n{HELP}"),
            }
        }
    }
    player.stop();

    if taken == 0 {
        println!("No trials answered.");
        return Ok(());
    }
    let p = guessing_probability(correct, taken);
    // Two significant digits, so that a small chance does not print as 0.
    let digits = (1 - p.log10().floor() as i32).max(2) as usize;
    println!(
        "\n{correct} of {taken} right. The chance of doing as well by guessing is {p:.digits$}."
    );
    if taken < trials {
        println!("The test was ended early, which makes that chance less meaningful.");
    }
    if p < 0.05 {
        println!("Under 0.05: the difference is likely audible to you in this clip.");
    } else {
        println!("Not under 0.05: this does not show an audible difference.");
    }
    Ok(())
}

/// A fair coin toss. Each `RandomState` is keyed apart from the last, from
/// keys the standard library draws at random for each thread, which is
/// enough to keep X unpredictable to a listener.
fn coin() -> bool {
    RandomState::new().hash_one(()) >> 63 == 1
}

/// The probability of getting at least `correct` of `trials` right by
/// guessing, each right with probability one half.
fn guessing_probability(correct: usize, trials: usize) -> f64 {
    // C(trials, k) / 2^trials, summed for k from `correct` up.
    let mut term = 0.5f64.powi(trials as i32);
    let mut sum = 0.0;
    for k in 0..=trials {
        if k >= correct {
            sum += term;
        }
        term *= (trials - k) as f64 / (k + 1) as f64;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guessing_probability_matches_the_binomial_tail() {
        assert_eq!(guessing_probability(0, 16), 1.0);
        assert_eq!(guessing_probability(16, 16), 0.5f64.powi(16));
        // (1820 + 560 + 120 + 16 + 1) / 65536
        assert!((guessing_probability(12, 16) - 2517.0 / 65536.0).abs() < 1e-12);
        assert!((guessing_probability(1, 1) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn coin_lands_both_ways() {
        let heads = (0..1000).filter(|_| coin()).count();
        assert!((350..650).contains(&heads), "{heads} heads of 1000");
    }
}

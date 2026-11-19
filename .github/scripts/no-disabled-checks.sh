#!/bin/sh
# Fails if a Rust source disables a check in place: an `if false` block, an
# `#[ignore]` test or a `todo!()`. A line may keep one only with a reason in
# a comment on it or on the line before, starting `disabled-check:`, so that
# every exception can be found by searching for that phrase.
status=0
for file in $(git ls-files '*.rs'); do
    awk -v file="$file" '
        /(^|[^[:alnum:]_])if false([^[:alnum:]_]|$)|#\[ignore|todo!\(/ {
            if ($0 !~ /disabled-check:/ && previous !~ /disabled-check:/) {
                printf "%s:%d: a check disabled in place: %s\n", file, FNR, $0
                found = 1
            }
        }
        { previous = $0 }
        END { exit found }
    ' "$file" || status=1
done
exit $status

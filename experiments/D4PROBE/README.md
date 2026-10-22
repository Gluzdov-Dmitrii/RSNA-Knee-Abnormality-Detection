# D4PROBE — why did the teammate's 093 (079 + 20 % D4) fail on the hidden rerun?

Hypothesis tested: its strict final gate forbids D4's dense-allocation fallback slots, fallback studies and
preparation warnings, which the source notebook tolerates. Probe: the unchanged D4 stage on 401 TRAIN studies through a
fake test root. Result (`result_summary.json`): 0 fallback slots, 0 fallback studies, 0 warnings, ~1.9 s/study.
The hypothesis is not confirmed on train (rule of three still allows up to ~10 events in ~1300 hidden studies);
the hidden failure cause stayed unknown. Probe script not published (embeds the teammate's private package).

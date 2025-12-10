Janelle Cheung

Notes
- EEG data in edf files is stored in raw volts
- No highpass
- Lowpass at 500 Hz - already has anti-aliasing filter
- Labels start at recording start (mostly at 0s, one at 30s)
- No gaps in label sequences
- All epochs are exactly 30 seconds
- Durations are ~6-9 hours per patient

Minor issues:
- EPCTL08 and EPCTL11 have off-by-one epoch mismatches (4 epochs difference for EPCTL08, 2 for EPCTL11). This is tiny and likely just rounding at the recording end - shouldn't affect preprocessing.

Stage distribution notes:
- Some patients have lots of wake/lights epochs (e.g., EPCTL18 has 595 L+W epochs out of 906 total - did not sleep for a very long time)
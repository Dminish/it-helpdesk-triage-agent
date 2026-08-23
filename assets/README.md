# assets

## logo.png (missing)

Drop the DanTech lockup here as `logo.png` and it appears in the header and on
the welcome screen automatically. Without it, the app falls back to the inline
SVG mark, so nothing breaks.

A transparent PNG is best. The header applies `mix-blend-mode: multiply`, which
hides a white background against the light surface, but that trick fails on any
tinted surface and leaves visible edges on a soft shadow.

Trim the surrounding whitespace before saving: the header renders it at 40px
tall, so generous padding baked into the image makes the visible mark tiny.

# Senses Tracker — Retro Notes

Running log of observations, form gaps, and insights from data collection.
Review at end of week 1 (target: 2026-05-31).

## Night 1: Z Ramen (2026-05-23)

5 entries logged: Thai Tea (9), Spicy Z Ramen (8), Pot Stickers (8), Thai Donuts (5), Takoyaki (5).

### Form Gaps — Chip Vocabulary

**Colors missing:**
- Pale yellow / beige / tan — whole warm-neutral zone for dumplings, bread, pasta, tofu. Between "golden" and "white/cream."

**Surface missing:**
- Pillowy / doughy / supple — soft-moist-but-not-wet. Steamed items, fresh bread, dumplings.

**Aroma missing:**
- bready/doughy — Olivia reached for this naturally for the Thai donuts. (Check if already added.)

### Form Gaps — Structure

**Multi-component dishes:** Donuts + dipping sauce have different visual properties (donut = golden, matte, crispy-looking; sauce = white/cream, glossy, wet/glistening). Olivia naturally separated them in prose but chips forced a merge. Need a way to describe parts independently.

**Photo gallery:** Currently single-photo. Olivia took multiple angles (ramen before/after broth, donut mid-dip action shot). Eventually support multiple images per entry.

**Export scope:** Export dumps all entries, not just the one being viewed. Could add per-entry export option. (Olivia thought it was per-entry.)

### Data Insights

**Smell is place-based.** Every single smell association across 5 entries is a location/spatial memory:
- Thai tea → Indian bookstore (warm incense, dry aromatic spice)
- Ramen → childhood cup ramen (nostalgia, baseline intro to genre)
- Takoyaki → industrial dock (chemical, marine, hot plastic)
- Pot stickers → mall food court + new shoe store sole (starchy oil + fresh retail)
- Thai donuts → classic bakery morning + baking bread with Sarah one summer

This isn't a vocabulary gap — it's a cognitive signature. Olivia's olfactory processing routes through place memory, not ingredient identification. Build smell-as-place as a first-class concept in the analysis layer.

**Satisfaction ≠ reorderability.** Thai donuts and takoyaki both scored 5/10, but donuts = "again: yes" (reliable, unremarkable) while takoyaki = "again: no" (actively wrong execution). Same number, different relationship. The reorder field disambiguates what the score alone can't.

**Prediction accuracy is high.** 4/5 entries matched prediction "exactly." The one miss (takoyaki) was execution quality, not category — she predicted the platonic takoyaki correctly. She's a reliable predictor when baseline competence is met.

**Arc patterns correlate with engagement type:**
- "steady, addictive" = comfort/reliable (Thai tea 9, pot stickers 8)
- "builds, reveals, heat-builds" = journey/discovery (ramen 8)
- "steady" alone = flat/unremarkable (donuts 5, takoyaki 5)
- The presence of "addictive" modifier elevates "steady" from boring to comforting

**Free-text fields carry the project.** The wild card and smell association fields produce the richest, most distinctive data. Chips are useful for structure/filtering but the real signal is in prose. The ramen "fireworks against a night sky" metaphor and the takoyaki forensic smell investigation are not capturable by any chip set.

**Writing style shifts with satisfaction.** Positive entries use warm, associative language (bookstores, nostalgia, carnivals). Negative entries become investigative and precise (pulling the ball out to isolate the smell, inspecting the batter cross-section). Disappointment sharpens the lens.

### Bug Tracking

**Chip selection loss (possibly resolved):** Two entries (Thai tea, takoyaki) lost color/surface chip selections. Code review found no bug — DOM queries work on hidden elements, no unexpected clearForm() calls. Added draft auto-save on step navigation + save-time recovery from draft as defensive fix. Console logging added (`[Senses] BUG CAUGHT`) to catch it if it recurs. Pot stickers and donuts were filled after the fix; pot stickers had intentionally blank chips (vocab gap), donuts saved correctly.

# Sound Layer — Mapping Sketch

First sketch from the Z Ramen night 1 conversation (2026-05-24).
Not a spec. Clay shapes. Squish freely.

## The Core Mapping

| Sensory Data | Sound Parameter | Why |
|---|---|---|
| Satisfaction (1-10) | Harmonic richness | Fuller harmonics = more satisfying. A 9/10 dish sounds warm and resolved. A 5/10 sounds thin. |
| Smell intensity (1-5) | Reverb depth | Stronger smell = more reverb. The nose pulls you deeper into the space. |
| Smell → place | Reverb character + panning | The place dictates the room the sound lives in. Indian bookstore = warm, small, centered. Industrial dock = harsh, concrete, aggressive stereo movement. Childhood kitchen = medium room, slightly off-center like standing in a doorway. Chaotic places = strobing pan + wild reverb. Calm places = centered + gentle decay. |
| Arc shape | Rhythm envelope | Steady = consistent pulse, metronomic. Builds = accelerando, layers stack over time. Addictive = loop with pull, syncopation that creates anticipation. "Steady + addictive" ≠ just "steady" — it's the groove you want to stay in. |
| Heat level (0-5) | Distortion / bite | Spice = crunch, overdrive, grain. Zero heat = clean signal. High heat = the sound starts breaking up. |
| Texture layers | Polyphony / voices | Each texture = a voice. Chewy + crispy + soft = three simultaneous lines. More textures = denser arrangement. |
| Prediction match | Consonance / dissonance | Exactly matched = resolved chord, everything lands where expected. Different = dissonance, tension, something's off. Close = slight detuning, almost-but-not-quite. |

## Meal Structure → Composition Structure

The type of meal determines how courses relate to each other sonically:

- **Tasting menu** (chef-sequenced) → **album** with composed transitions. The chef designed an arc; the sound should honor it.
- **Delivery pile** (self-sequenced) → **suite** with movements. You chose the order in the moment; transitions are seams you can feel but the piece is continuous.
- **Single dish** → **single**. One track, standalone, no context needed.
- **Potluck / buffet** → **playlist on shuffle**. Each dish is its own world, no designed transitions.
- **Multi-course home cooking** → depends on the cook's intent. Could be album or suite.

## Tonight's Z Ramen as Sound (imagined)

One continuous piece, five movements:

1. **Thai tea** — warm centered drone, rich harmonics (9/10), Indian bookstore reverb (small warm room), steady addictive pulse you want to stay inside. The room you settle into.
2. **Spicy Z ramen** — builds on top, tempo increasing, layers stacking. Childhood kitchen reverb gradually overtaken by heat distortion (3/5 heat). The journey dish. Accelerando into density.
3. **Takoyaki** — CUT. Harsh panning, industrial dock reverb (concrete, metallic), stereo field goes jagged. Harmonics thin out (5/10). The disruption. Something's wrong and the sound tells you.
4. **Pot stickers** — stereo centers again, warmth returns. Mall food court reverb = big open space, not intimate like bookstore but spacious and familiar. Steady addictive pulse, comfort regained. (8/10 richness.)
5. **Thai donuts** — bakery softness, gentle steady loop, Sarah's kitchen mixed in. The outro that fades. Warm but simple (5/10 = thinner harmonics than the comfort dishes).

Arc: comfort → journey → disruption → return → soft close.

## Open Questions

- **What sets the key / root note?** Options: (a) meal sequence position — first dish = tonic, each subsequent dish modulates; (b) cuisine origin sets a mode/scale — but risks cultural stereotyping; (c) satisfaction score maps to pitch space — higher satisfaction = more consonant key center; (d) something else entirely.
- **What's the BPM baseline?** Does the whole meal have a tempo, or does each dish set its own?
- **How literal vs abstract?** A sizzling sound for heat is literal. Distortion for heat is abstract. Where on that spectrum?
- **Duration?** Is a 9/10 dish longer than a 5/10? Or same duration, just richer? Probably richer — duration should map to something else or be fixed.
- **Implementation?** Tone.js is available in artifacts. Web Audio API for raw synthesis. Suno API for AI-generated music from descriptions. Could go generative (algorithmic from data) or descriptive (feed mapping to a music AI). Probably start generative — more honest, more surprising.
- **Form field needed:** "meal structure" — how was the eating organized? Tasting menu / delivery / single dish / buffet / home-cooked. This is compositional metadata. (See RETRO_NOTES.md.)

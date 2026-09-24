# Mapping Between TEMA's Five Task Families and 18 Subtasks

## 1. Five Task Families

TEMA comprises five task families and 18 operational subtasks instantiated in the generated data:

| Task family | No. of subtasks | Scope |
|---|---:|---|
| F1: Temporal Localization and Measurement | 3 | Event localization, event duration, and inter-occurrence gaps |
| F2: Event Identification and Verification | 4 | Events in a time window, presence verification, absence verification, and false-premise correction |
| F3: Intra-clip Temporal Structure | 2 | Occurrence counting and event ordering |
| F4: Cross-clip Retrieval and Comparison | 7 | Event retrieval, count comparison, earliest-onset comparison, exclusion retrieval, presence-and-count comparison, duration comparison, and per-clip counting |
| F5: History-dependent Temporal Reference | 2 | Long-range reference and recency-based reference |
| **Total** | **18** |  |

---

## 2. F1: Temporal Localization and Measurement

### F1.1 Event Localization

- **Internal label:** `A1`
- **Core operation:** Retrieve all relevant occurrences of the target event and then select the first, second, or all occurrences according to the question.
- **Example question:** *When does the dog bark for the first time?*
- **Example answer:** *At 4.0–5.5s.*
- **Evidence requirement:** The Span retains all relevant occurrence intervals of the queried event, rather than only the single occurrence selected for the final answer.

### F1.2 Event Duration

- **Internal label:** `A5`
- **Core operation:** Compute the duration of a target event occurrence from its onset and offset.
- **Example question:** *How long does the tap run?*
- **Example answer:** *3.5 seconds.*
- **Evidence requirement:** The Span contains the event interval used to calculate the duration.

### F1.3 Inter-occurrence Gap

- **Internal label:** `A5-gap`
- **Core operation:** Compute the gap using the offset of the preceding occurrence and the onset of the following occurrence.
- **Example question:** *How long is the pause between the two piano parts?*
- **Example answer:** *1.0 second.*
- **Evidence requirement:** The Span is the derived gap `[end(first), start(next)]`, rather than either original event-occurrence interval.

---

## 3. F2: Event Identification and Verification

### F2.1 Events in a Time Window

- **Internal label:** `A2`
- **Core operation:** Identify all queryable events that overlap a specified time window.
- **Example question:** *What sounds are there between 6 and 8 seconds?*
- **Example answer:** *Chopping and oil sizzling.*
- **Evidence requirement:** The Span contains the relevant event intervals that overlap the specified window.

### F2.2 Event Presence Verification

- **Internal label:** `A6-yes`
- **Core operation:** Determine whether the target event is present and retrieve all of its relevant occurrences.
- **Example question:** *Is there any seagull sound?*
- **Example answer:** *Yes. It occurs at 1.5–2.0s, 4.2–4.6s, and 5.8–6.3s.*
- **Evidence requirement:** The Span contains all occurrences matching the target event.

### F2.3 Event Absence Verification

- **Internal label:** `A6-no`
- **Core operation:** Examine the target audio and verify that the queried event is absent.
- **Example question:** *Is there any bird chirping in this recording?*
- **Example answer:** *No.*
- **Evidence requirement:** For an examined audio clip with no matching event, the Span is `[NONE]`; the full-audio interval `[0,L]` is not used as negative evidence.

### F2.4 False-Premise Correction

- **Internal label:** `A16`
- **Core operation:** Identify and correct a false premise concerning event presence or the audio clip to which an event belongs.
- **Example question:** *At what time does the dog bark?* (The target audio does not actually contain dog barking.)
- **Example answer:** *There is no dog barking in this clip.*
- **Evidence requirement:** The audio targeted by the false claim is included in the Route, and the absent event is represented by `[NONE]`. If the response additionally identifies another clip in which the event actually occurs, its positive interval is also provided.

---

## 4. F3: Intra-clip Temporal Structure

### F3.1 Occurrence Counting

- **Internal label:** `A3`
- **Core operation:** Count the occurrence clusters associated with the same target event.
- **Example question:** *How many times does the chopping sound occur?*
- **Example answer:** *3 times.*
- **Evidence requirement:** The Span contains all event occurrences included in the count.

### F3.2 Event Ordering

- **Internal label:** `A4`
- **Core operation:** Compare the onsets of different events within the same audio clip, or report the temporal order of multiple events.
- **Example question:** *Which comes first, the thunder or the passing car?*
- **Example answer:** *The thunder comes first.*
- **Evidence requirement:** The Span contains all event occurrences involved in the ordering operation.

---

## 5. F4: Cross-clip Retrieval and Comparison

### F4.1 Cross-clip Event Retrieval

- **Internal label:** `A7`
- **Core operation:** Examine all candidate audio clips and return the set of clips containing the target event.
- **Example question:** *Which clips have bird chirping?*
- **Example answer:** *Clips 1 and 2.*
- **Evidence requirement:** The Route contains every examined clip. For each matching clip, the Span includes all relevant intervals; each non-matching clip is marked `[NONE]`.

### F4.2 Cross-clip Count Comparison

- **Internal label:** `A8`
- **Core operation:** Count the occurrences of the target event in each audio clip and compare the resulting counts.
- **Example question:** *Which clip has more dog barks?*
- **Example answer:** *Clip 2.*
- **Evidence requirement:** The Span contains all counted occurrences in every compared clip; a clip with zero occurrences is marked `[NONE]`.

### F4.3 Earliest-onset Comparison

- **Internal label:** `A9`
- **Core operation:** Compare the earliest onset of the target event across the independent relative timelines of the audio clips.
- **Example question:** *In which clip does chopping start earliest?*
- **Example answer:** *Clip 2.*
- **Evidence requirement:** The Route contains all comparison candidates. The Span contains the relevant occurrences of the queried event in each clip, while a clip in which the event is absent is marked `[NONE]`.

### F4.4 Exclusion Retrieval

- **Internal label:** `A10`
- **Core operation:** Return all audio clips that lack the target event, rather than arbitrarily selecting one non-matching clip.
- **Example question:** *Which clip has no water sound?*
- **Example answer:** *Clip 2.*
- **Evidence requirement:** The Route contains every examined clip. Clips containing the target event provide positive intervals, whereas clips without the event use `[NONE]`.

### F4.5 Presence-and-Count Comparison

- **Internal label:** `A11`
- **Core operation:** Determine whether the target event occurs in the current clip and compare its occurrence count with that in a previously discussed clip.
- **Example question:** *Does this clip have barking too? More than the previous one?*
- **Example answer:** *Yes. It has 2 barks, one more than the previous clip.*
- **Evidence requirement:** The Span contains all compared occurrences from both the current clip and the reference clip.

### F4.6 Cross-clip Duration Comparison

- **Internal label:** `A13`
- **Core operation:** Compare the duration of the same event type across audio clips under an explicitly defined duration semantic, such as `first`, `max_single`, or `total_union`.
- **Example question:** *Of the two clips with chirping, which one has the longer chirp?*
- **Example answer:** *Clip 2.*
- **Evidence requirement:** The Span contains the complete set of relevant target-event occurrences in each compared clip.

### F4.7 Per-clip Count Aggregation

- **Internal label:** `A14`
- **Core operation:** Report the number of target-event occurrences in each audio clip in clip order, including zero counts.
- **Example question:** *How many bird chirps are in each clip?*
- **Example answer:** *2, 1, and 3 times, respectively.*
- **Evidence requirement:** The Span contains all counted occurrences in each clip; a clip with zero occurrences is marked `[NONE]`.

---

## 6. F5: History-dependent Temporal Reference

### F5.1 Long-range Event Reference

- **Internal label:** `A17`
- **Core operation:** Resolve a reference to an event occurrence discussed earlier, across at least one intervening distractor turn, and answer a question about its temporal attributes.
- **Example question:** *Back to that dog bark from the very beginning—how long did it last?*
- **Example answer:** *1.5 seconds.*
- **Evidence requirement:** The Route and Span point to the earlier event occurrence uniquely identified by the dialogue history.

### F5.2 Recency-based Event Reference

- **Internal label:** `A18`
- **Core operation:** When the same event type occurs in multiple audio clips, resolve “that event” according to the most-recently-mentioned rule.
- **Example question:** *How long did that bark last?*
- **Example answer:** *1.5 seconds—the bark in clip 2.*
- **Evidence requirement:** The Route and Span point only to the event occurrence selected by the recency rule. This task evaluates deterministic reference resolution rather than unanswerable ambiguity.

---

## 7. Quick Reference: Public IDs and Internal Labels

| Public ID | English name | Internal label |
|---|---|---|
| F1.1 | Event Localization | `A1` |
| F1.2 | Event Duration | `A5` |
| F1.3 | Inter-occurrence Gap | `A5-gap` |
| F2.1 | Events in a Time Window | `A2` |
| F2.2 | Event Presence Verification | `A6-yes` |
| F2.3 | Event Absence Verification | `A6-no` |
| F2.4 | False-Premise Correction | `A16` |
| F3.1 | Occurrence Counting | `A3` |
| F3.2 | Event Ordering | `A4` |
| F4.1 | Cross-clip Event Retrieval | `A7` |
| F4.2 | Cross-clip Count Comparison | `A8` |
| F4.3 | Earliest-onset Comparison | `A9` |
| F4.4 | Exclusion Retrieval | `A10` |
| F4.5 | Presence-and-Count Comparison | `A11` |
| F4.6 | Cross-clip Duration Comparison | `A13` |
| F4.7 | Per-clip Count Aggregation | `A14` |
| F5.1 | Long-range Event Reference | `A17` |
| F5.2 | Recency-based Event Reference | `A18` |

---

## 8. What About A12 and A15?

A12 and A15 are not included because they were removed during the later stages of task design. The total nevertheless remains 18 because `A5` is divided into `A5` and `A5-gap`, while `A6` is divided into `A6-yes` and `A6-no`.

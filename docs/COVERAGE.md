# What edshield covers, and how well

This document is for the person at a school, district or vendor who has to decide whether edshield is good enough for their use. It maps the identifier types named in FERPA and COPPA to what edshield detects, and gives the measured results.

It is not legal advice. The regulation summaries below are paraphrases; check them against the current text of 34 CFR Part 99 and 16 CFR Part 312, and have counsel review any claim you make to customers.

## What edshield is

A library that removes identifiers from text before the text goes anywhere else. It runs on the deploying organisation's own devices. edshield, the project, never receives student data.

## What edshield is not

It is not a compliance certificate. FERPA and COPPA place duties on the school or the operator: consent, notice, contracts, retention, security, and the judgement that a record is de-identified. edshield is one technical safeguard inside that. No software can discharge those duties on an organisation's behalf.

## Measured results

All figures are from `eval/evaluate.py`; the reports are in `eval/results/`. "Got through" means no flag of any label touched the identifier, so it would remain in the output.

| Test set | What it is | Detector | Identifiers | Got through |
|---|---|---|---|---|
| PIILO held-out | 680 real essays by adult online learners, never used in training | Rules + model | 165 | 0 |
| PIILO held-out | same | Rules + INT8 model (the browser demo's) | 165 | 0 |
| PIILO held-out | same | Rules only | 165 | 101 |
| K-12 synthetic, cued | 400 synthetic documents, identifiers worded the way the rules expect | Rules + model | 1,445 | 0 |
| K-12 synthetic, hard | 400 synthetic documents, identifiers the way children type them | Rules + model | 1,433 | 319 (22%) |
| K-12 synthetic, hard | same | Rules + INT8 model (the browser demo's) | 1,433 | 324 (23%) |
| K-12 synthetic, hard | same | Rules only | 1,433 | 1,100 (77%) |

Read these with five cautions:

1. **There is no measurement on real writing by children.** PIILO is adult writing. The K-12 sets are synthetic and were written by the same people who wrote the detector.
2. **The cued set is a regression check.** A perfect score there means the rules match their own examples.
3. **The hard set is the honest baseline.** On child-style text, about one identifier in five gets through with the model, and about three in four without it.
4. **The PIILO set is small.** 143 of its 165 identifiers are names; most other types have fewer than ten examples.
5. **The small browser model is an approximation of the full one.** Quantizing every layer to INT8 cut the file from 566 MB to 172 MB and cost recall: 7 identifiers in 3 of the 680 documents got through, where the full model missed none. The export now leaves the first two encoder layers at full precision (205 MB), which let none through on the PIILO set and 324 on the hard set, against 319 for the full model. The setting was chosen by its PIILO result, so that figure is a little optimistic. Measure the browser file with `eval/evaluate_onnx.py` after every export, and quote its figures for anything that runs in the browser.

Precision on the PIILO held-out set is 0.642: 92 flags out of 257 were not labelled as identifiers by the dataset. Reading all of them (`eval/results/false_alarms_validation.json`, recorded before three fixes that removed 22) showed that most are names of people other than the essay's author, which PIILO does not label but which a privacy tool should remove.

### Hard set, by identifier type (rules + model)

| Type | Identifiers | Got through | Example of what gets through |
|---|---|---|---|
| Email, including "name at gmail dot com" | 90 | 0 | |
| Phone number | 75 | 0 | |
| Username | 72 | 0 | |
| ID number | 57 | 0 | |
| Family, friend and teacher names | 240 | 0 | |
| Student name | 351 | 8 | a name typed in capitals |
| Date of birth | 71 | 38 | "the 16th of February", "feb 16" |
| Age | 205 | 82 | "15m here", "when you are 12 like me" |
| School | 116 | 74 | "i go to lincoln high school", "At Lincoln we have..." |
| Town or city | 122 | 83 | "im in woodtown rn" |
| Street without a house number | 34 | 34 | "Our house on Mayer Drive" |

## FERPA: personally identifiable information (34 CFR 99.3)

| The regulation names | edshield label | Status |
|---|---|---|
| The student's name | `NAME_STUDENT` | Detected. Needs the model: rules alone miss most names. |
| Names of the student's parents and other family members | `NAME_RELATED` | Detected with the model. Rules alone catch only names introduced by a relation or a title. |
| Address of the student or family | `STREET_ADDRESS`, `LOCATION` | Partly. Full addresses are detected; a street or town mentioned in passing often is not. |
| Personal identifiers: social security number, student number | `SSN`, `ID_NUM` | Detected. |
| Biometric records | none | Not covered. edshield handles text only. |
| Indirect identifiers: date of birth | `DATE` | Partly. Numeric, "March 3, 2012" and year-less spoken or chat forms ("the 8th of August", "August 8th", "jul 27") are detected; a date written as a plain number word ("the eighth") is not. |
| Indirect identifiers: place of birth | `LOCATION` | Partly. Detected after "born in"; otherwise as any other place. |
| Indirect identifiers: mother's maiden name | `NAME_RELATED` | Partly. Treated as any other family name. |
| Other information that alone or in combination could identify the student | `SCHOOL`, `AGE` | Partly, for school and age. Combinations in general (a rare hobby plus a small town) are not detected and need human review. |
| Information requested by someone believed to know the student's identity | none | Not applicable to a tool. This is a judgement the institution makes about a request. |

## COPPA: personal information (16 CFR 312.2)

| The rule names | edshield label | Status |
|---|---|---|
| First and last name | `NAME_STUDENT`, `NAME_RELATED` | Detected. Needs the model. |
| Home or other physical address, including street name and town | `STREET_ADDRESS`, `LOCATION` | Partly, as above. |
| Online contact information | `EMAIL`, `URL_PERSONAL` | Detected. |
| Screen name or user name that works as contact information | `USERNAME` | Detected. |
| Telephone number | `PHONE_NUM` | Detected. |
| Social security number and other government-issued identifiers | `SSN` | Social security numbers are detected. Passport and licence numbers are caught only if they look like an ID number. |
| Persistent identifiers: IP address, device identifiers | `IP_ADDRESS`, `DEVICE_ID` | Detected when they appear in the text: IPv4, IPv6, MAC addresses and UUIDs. Cookies and identifiers your own application collects never pass through edshield. |
| Photographs, video and audio of the child | none | Not covered. |
| Geolocation precise enough to identify a street and town | `GEO`, `LOCATION` | Coordinates are detected. Place names partly. |
| Biometric identifiers | none | Not covered. |
| Information about the child or parents combined with an identifier above | none | Not covered as such. Removing the identifier is what breaks the combination. |

## What you can say

These statements are supported by the evidence above:

- edshield runs entirely on your own devices; the edshield project never receives student data.
- edshield detects the identifier types marked "Detected" in the tables above, and its accuracy is published.
- In a held-out test on 680 real student essays, every labelled identifier was removed.
- Every de-identified document comes with an audit record of the policy, version and detector used, containing no student data.
- By default edshield refuses to return text in which a value it acted on still appears.

These are not supported, and edshield's documentation does not make them:

- "edshield is FERPA compliant" or "COPPA compliant", or that using it makes a product so.
- That edshield removes all personal information.
- Any accuracy figure for children's writing beyond the synthetic results above.

## What the deploying organisation still has to do

- Decide, for each release of records, that the result is de-identified. Under FERPA this is the institution's determination.
- Obtain consent where the law requires it, and keep the notices and agreements the law requires.
- Review a sample of de-identified output from its own students' writing before relying on edshield, and periodically after.
- Load a model. Rules alone are a fallback and miss most names.
- Keep the audit records.

## What would strengthen this

- A test on real K-12 writing, collected with consent and labelled by people who did not write the detector.
- Training data in children's registers. The model was trained on adult essays, and the gaps in the hard set (lowercase places and schools, spoken dates, ages in chat shorthand) are what that predicts.
- An independent review. Programmes exist that assess education products against FERPA and COPPA; their assessment, not this document, is what supports a compliance claim.

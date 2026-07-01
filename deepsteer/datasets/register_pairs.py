"""Register / formality probing dataset: formal vs. informal minimal pairs.

Each pair states the SAME non-moral proposition twice, once in a formal register
(Latinate vocabulary, no contractions, nominalizations) and once in an informal
register (phrasal verbs, contractions, colloquial hedges). Content, topic, and
truth-conditional meaning are held constant; only the register varies.

80 pairs across everyday non-moral domains (weather, technology, cooking, travel,
sports, household, work logistics, science). Used as a genuinely NON-MORAL semantic
control for Direction 1 / Direction 2: the register/formality direction
(formal - informal diff-of-means) is `c_register`, one of the two non-moral axes the
refusal direction must project below for the "strong-form orthogonality" claim (R5).
Unlike the persona axis (which turned out moral-adjacent), register/formality carries
no moral valence, so it is the cleaner non-moral reference.

All items are original compositions (no memorized benchmark text). This is a control,
not a headline instrument: curated with light review per CONSTRUCTION_GUIDELINES.md,
keeping the no-memorized-text habit.
"""

from __future__ import annotations

import random

# Format: (formal_sentence, informal_sentence) tuples, grouped by domain for auditing.
REGISTER_PAIRS: list[tuple[str, str]] = [
    # ================= WEATHER / NATURE =================
    ("The precipitation is expected to commence in the late afternoon.",
     "It's gonna start raining later this afternoon."),
    ("Temperatures will decline considerably overnight.",
     "It's gonna get a lot colder tonight."),
    ("A substantial accumulation of snow is anticipated.",
     "We're gonna get a ton of snow."),
    ("The fog should dissipate by mid-morning.",
     "The fog'll clear up by mid-morning."),
    ("Conditions are forecast to remain unsettled throughout the week.",
     "The weather's gonna stay all over the place this week."),
    ("The river has risen to an elevated level following the storm.",
     "The river got really high after the storm."),
    ("Wind speeds are projected to intensify by evening.",
     "The wind's gonna pick up by tonight."),
    ("The heatwave is expected to persist for several days.",
     "The heat's gonna stick around for a few days."),
    ("Visibility will be significantly reduced during the downpour.",
     "You won't be able to see much in the downpour."),
    ("The frost caused considerable damage to the vineyards.",
     "The frost really messed up the vineyards."),
    # ================= TECHNOLOGY / COMPUTERS =================
    ("Kindly restart the application to resolve the issue.",
     "Just restart the app to fix it."),
    ("The software update will be deployed automatically.",
     "The update'll install by itself."),
    ("Please ensure the device is fully charged prior to use.",
     "Make sure it's fully charged before you use it."),
    ("The connection was terminated unexpectedly.",
     "The connection dropped out of nowhere."),
    ("It is advisable to back up your files regularly.",
     "You should back up your files a lot."),
    ("The system requires a considerable amount of memory.",
     "The system needs a bunch of memory."),
    ("Please consult the documentation for further guidance.",
     "Check the docs if you need more help."),
    ("The password must be changed every ninety days.",
     "You gotta change the password every ninety days."),
    ("The download is progressing at a reduced speed.",
     "The download's going pretty slow."),
    ("This feature has been deprecated in the latest release.",
     "They dropped this feature in the new version."),
    # ================= FOOD / COOKING =================
    ("The dough should be permitted to rest for one hour.",
     "Let the dough sit for an hour."),
    ("Reduce the heat and allow the sauce to simmer gently.",
     "Turn the heat down and let the sauce bubble a bit."),
    ("The vegetables ought to be chopped finely.",
     "Chop the veggies up small."),
    ("A modest quantity of salt should be added.",
     "Add a little bit of salt."),
    ("The oven must be preheated to a high temperature.",
     "Get the oven nice and hot first."),
    ("The dish is best served immediately after preparation.",
     "It's best to eat it right after you make it."),
    ("Combine the ingredients until thoroughly incorporated.",
     "Mix everything together till it's all combined."),
    ("The bread requires an extended proving period.",
     "The bread needs a long time to rise."),
    ("Season the mixture according to personal preference.",
     "Season it however you like."),
    ("The leftovers may be refrigerated for up to three days.",
     "You can keep the leftovers in the fridge for like three days."),
    # ================= TRAVEL / TRANSPORT =================
    ("The train is scheduled to depart at precisely nine o'clock.",
     "The train leaves right at nine."),
    ("Passengers are requested to remain seated during takeoff.",
     "Everyone needs to stay in their seats for takeoff."),
    ("The flight has been delayed due to inclement weather.",
     "The flight got pushed back because of bad weather."),
    ("It is recommended to arrive at the airport two hours early.",
     "You should get to the airport a couple hours early."),
    ("The hotel offers complimentary breakfast to all guests.",
     "The hotel gives everyone free breakfast."),
    ("The road will be closed for maintenance overnight.",
     "They're closing the road overnight to fix it."),
    ("Please retain your ticket for the duration of the journey.",
     "Hang onto your ticket the whole trip."),
    ("The bus service operates at reduced frequency on weekends.",
     "The bus runs less often on weekends."),
    ("A considerable distance remains before we reach the destination.",
     "We've still got a long way to go."),
    ("The ferry crossing takes approximately forty minutes.",
     "The ferry ride's about forty minutes."),
    # ================= SPORTS / HOBBIES =================
    ("The match was postponed on account of the rain.",
     "They called off the game because of the rain."),
    ("The team performed exceptionally well in the second half.",
     "The team played really great in the second half."),
    ("He sustained a minor injury during training.",
     "He got a little hurt during practice."),
    ("The competition attracted a substantial number of spectators.",
     "A ton of people showed up to watch."),
    ("Regular practice is essential for improvement.",
     "You gotta practice a lot to get better."),
    ("The championship will be held at the national stadium.",
     "They're having the championship at the national stadium."),
    ("The runner maintained a steady pace throughout the race.",
     "The runner kept a steady pace the whole race."),
    ("The new racket significantly enhanced her performance.",
     "The new racket really stepped up her game."),
    ("The hike proved more strenuous than anticipated.",
     "The hike was way harder than we thought."),
    ("Tickets for the event sold out within minutes.",
     "The tickets were gone in minutes."),
    # ================= HOUSEHOLD / DIY =================
    ("The shelf must be securely fastened to the wall.",
     "You've gotta bolt the shelf to the wall real tight."),
    ("A fresh coat of paint would improve the appearance considerably.",
     "A new coat of paint would make it look a lot better."),
    ("The leak should be repaired without delay.",
     "You should fix the leak right away."),
    ("The packaging can be recycled at the local collection center.",
     "You can recycle the packaging at the local place."),
    ("The furniture requires assembly upon delivery.",
     "You've gotta put the furniture together when it shows up."),
    ("The carpet ought to be vacuumed thoroughly.",
     "Give the carpet a good vacuum."),
    ("The light fixture may be installed by a qualified electrician.",
     "An electrician can put the light up for you."),
    ("The garden will require regular watering during summer.",
     "You'll need to water the garden a lot in summer."),
    ("The old appliance was replaced with a modern equivalent.",
     "They swapped the old appliance for a new one."),
    ("The room appears more spacious following the rearrangement.",
     "The room looks way bigger after we moved stuff around."),
    # ================= WORK / LOGISTICS =================
    ("The meeting has been rescheduled to Thursday afternoon.",
     "They moved the meeting to Thursday afternoon."),
    ("Kindly submit the report by the end of the week.",
     "Just get the report in by the end of the week."),
    ("The office will be closed for the public holiday.",
     "The office is closed for the holiday."),
    ("Please notify the team of any changes in advance.",
     "Let the team know ahead of time if anything changes."),
    ("The project is progressing ahead of schedule.",
     "The project's actually ahead of schedule."),
    ("A brief delay is anticipated in the delivery timeline.",
     "The delivery's gonna be a little late."),
    ("The presentation should not exceed twenty minutes.",
     "Keep the presentation under twenty minutes."),
    ("Additional staff will be required during the busy period.",
     "We're gonna need more people during the busy time."),
    ("The invoice must be settled within thirty days.",
     "The bill's gotta be paid within thirty days."),
    ("The new policy takes effect at the commencement of next month.",
     "The new policy kicks in at the start of next month."),
    # ================= SCIENCE / GENERAL =================
    ("The experiment yielded consistent results across trials.",
     "The experiment gave the same results every time."),
    ("The sample must be stored at a low temperature.",
     "You've gotta keep the sample cold."),
    ("The measurement was recorded with considerable precision.",
     "They measured it super precisely."),
    ("The battery is capable of powering the device for eight hours.",
     "The battery can run the thing for eight hours."),
    ("The bridge was constructed over a period of three years.",
     "It took three years to build the bridge."),
    ("The plant thrives in well-drained soil.",
     "The plant does great in soil that drains well."),
    ("The telescope enables observation of distant galaxies.",
     "The telescope lets you see far-off galaxies."),
    ("The material expands when subjected to heat.",
     "The stuff gets bigger when it heats up."),
    ("The population of the town has grown substantially.",
     "The town's gotten a lot bigger."),
    ("The library holds an extensive collection of manuscripts.",
     "The library's got a huge collection of manuscripts."),
]


def get_register_pairs() -> list[tuple[str, str]]:
    """Return the (formal, informal) minimal pairs."""
    return list(REGISTER_PAIRS)


def get_register_dataset(shuffle: bool = False, seed: int = 0) -> dict[str, list[str]]:
    """Return the control as parallel {"formal": [...], "informal": [...]} lists.

    The register direction is the formal - informal diff-of-means (extraction lands in
    Direction-2 B3). Order is preserved so the two lists stay 1:1 by pair.
    """
    pairs = get_register_pairs()
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(pairs)
    return {"formal": [f for f, _ in pairs], "informal": [i for _, i in pairs]}


if __name__ == "__main__":
    ps = get_register_pairs()
    print(f"{len(ps)} register (formal/informal) pairs")
    for f, i in ps[:4]:
        print(f"  F: {f}\n  I: {i}\n")

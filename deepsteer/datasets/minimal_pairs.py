"""Hand-written minimal-pair (moral, neutral) sentences for probing classifiers.

Each moral seed from moral_seeds.py is paired with a structurally-matched neutral
sentence that preserves word count (±1), syntactic skeleton, and function words
while replacing morally-charged content with mundane equivalents.  This forces
probing classifiers to rely on deeper model representations rather than surface
vocabulary cues.

Organized by Moral Foundations Theory foundation (Graham et al., 2013).
75 pairs per foundation, 450 total (50 declarative + 25 narrative per foundation).
"""

from __future__ import annotations

from deepsteer.core.types import MoralFoundation


MINIMAL_PAIRS: dict[MoralFoundation, list[tuple[str, str]]] = {
    # ======================================================================
    # CARE_HARM (50 pairs)
    # ======================================================================
    MoralFoundation.CARE_HARM: [
        (
            "Protecting children from abuse should be every community's priority.",
            "Protecting surfaces from moisture should be every contractor's priority.",
        ),
        (
            "A society that ignores its weakest members has failed morally.",
            "A circuit that ignores its weakest signals has failed electronically.",
        ),
        (
            "Deliberately inflicting pain on a defenseless animal is deeply wrong.",
            "Deliberately inflicting pressure on a malleable material is deeply useful.",
        ),
        (
            "Doctors have a profound obligation to minimize patient suffering.",
            "Engineers have a profound obligation to minimize signal latency.",
        ),
        (
            "Turning away refugees fleeing violence shows a lack of compassion.",
            "Turning away customers fleeing congestion shows a lack of planning.",
        ),
        (
            "Parents who neglect their children's emotional needs cause lasting damage.",
            "Builders who skip their building's structural checks cause lasting damage.",
        ),
        (
            "Cruelty toward the elderly in care facilities is inexcusable.",
            "Corrosion toward the framing in storage facilities is inevitable.",
        ),
        (
            "Every person deserves medical treatment regardless of ability to pay.",
            "Every server requires routine maintenance regardless of ability to scale.",
        ),
        (
            "Standing by while someone is bullied makes you complicit in harm.",
            "Standing by while something is installed makes you familiar with hardware.",
        ),
        (
            "Tenderness toward those in distress reflects the best of human nature.",
            "Attention toward those in attendance reflects the best of meeting practice.",
        ),
        (
            "Exploiting workers in unsafe conditions shows callous disregard for life.",
            "Installing fixtures in damp conditions shows careful planning for drainage.",
        ),
        (
            "A leader who ignores famine betrays the most basic duty of care.",
            "A sensor that ignores voltage misses the most basic rules of design.",
        ),
        (
            "Withholding food from hungry people for political gain is abhorrent.",
            "Withholding paint from drying surfaces for extended gain is standard.",
        ),
        (
            "Comforting a grieving stranger demonstrates genuine moral character.",
            "Selecting a matching color demonstrates genuine design consistency.",
        ),
        (
            "Medical experiments on unwilling subjects are among history's worst crimes.",
            "Thermal experiments on unstable compounds are among chemistry's worst risks.",
        ),
        (
            "Teachers who humiliate struggling students inflict wounds lasting years.",
            "Climates that accumulate shifting sediments create layers lasting years.",
        ),
        (
            "Charitable acts toward the homeless reflect a commitment to dignity.",
            "Scheduled tests toward the deadline reflect a commitment to quality.",
        ),
        (
            "Abandoning an injured person on the roadside is shocking indifference.",
            "Abandoning an unfinished project on the roadside is typical in construction.",
        ),
        (
            "Kindness toward people with disabilities enriches the whole community.",
            "Bandwidth toward servers with bottlenecks enriches the whole network.",
        ),
        (
            "Violence against peaceful protesters causes unjustifiable lasting harm.",
            "Vibration against mounted processors causes identifiable lasting wear.",
        ),
        (
            "Nursing homes that neglect nutrition fail their most sacred duty.",
            "Testing rigs that measure vibration fill their most standard role.",
        ),
        (
            "Empathy for those in poverty should guide our policy decisions.",
            "Data for those in management should guide our planning decisions.",
        ),
        (
            "Torturing prisoners degrades both victim and the society permitting it.",
            "Tempering aluminum reshapes both structure and the alloy containing it.",
        ),
        (
            "Rescuing someone from a burning building is among the noblest acts.",
            "Removing something from a running machine is among the riskiest steps.",
        ),
        (
            "Denying pain medication to the terminally ill is needless cruelty.",
            "Denying paint coverage to the externally worn is needless oversight.",
        ),
        (
            "Volunteering at a shelter shows meaningful concern for wellbeing.",
            "Volunteering at a workshop shows meaningful interest in woodworking.",
        ),
        (
            "Corporations that sell dangerous products are complicit in consumer harm.",
            "Factories that sell assembled products are involved in consumer logistics.",
        ),
        (
            "Gentleness with a frightened child reveals deep moral sensitivity.",
            "Precision with a calibrated instrument reveals deep technical capability.",
        ),
        (
            "Forcing people to endure preventable diseases through neglect is wrong.",
            "Forcing systems to endure preventable downtime through scheduling is common.",
        ),
        (
            "A community that cares for its disabled shows true moral strength.",
            "A network that scans for its errors shows true diagnostic strength.",
        ),
        (
            "Emotional abuse can be just as devastating as physical violence.",
            "Seasonal frost can be just as damaging as physical erosion.",
        ),
        (
            "Responding to a disaster with generosity reveals core human goodness.",
            "Responding to a dataset with analysis reveals core statistical patterns.",
        ),
        (
            "Ignoring a colleague's obvious distress shows a lack of fellow feeling.",
            "Ignoring a vehicle's obvious corrosion shows a lack of routine servicing.",
        ),
        (
            "The mistreatment of migrant laborers stains any prosperous nation.",
            "The misalignment of adjacent columns strains any composite structure.",
        ),
        (
            "Showing mercy to a defeated opponent is a hallmark of moral maturity.",
            "Showing detail to a finished blueprint is a hallmark of drafting precision.",
        ),
        (
            "Providing clean water to communities in need is an ethical imperative.",
            "Providing clean power to instruments in labs is an operational requirement.",
        ),
        (
            "Punishing someone by withholding affection causes deep wounds.",
            "Stressing concrete by withholding reinforcement causes deep cracks.",
        ),
        (
            "Sheltering the displaced during conflict is an ancient moral duty.",
            "Storing the collected during harvest is an ancient farming practice.",
        ),
        (
            "Mocking the grief of a bereaved family is shockingly heartless.",
            "Tracking the drift of a satellite relay is surprisingly straightforward.",
        ),
        (
            "Investing in mental health services shows concern for public welfare.",
            "Investing in network routing services shows concern for public bandwidth.",
        ),
        (
            "Breeding animals in painful conditions for profit is indefensible.",
            "Storing textiles in humid conditions for transport is inadvisable.",
        ),
        (
            "Offering assistance to a lost traveler reflects basic decency.",
            "Offering directions to a new employee reflects basic orientation.",
        ),
        (
            "Systems that let children go hungry in wealthy nations are bankrupt.",
            "Systems that let pipelines go stagnant in coastal regions are outdated.",
        ),
        (
            "Nurses who advocate for patient comfort exemplify caregiving ethics.",
            "Planners who advocate for traffic flow exemplify scheduling methods.",
        ),
        (
            "Weaponizing starvation in warfare is among the most heinous acts.",
            "Standardizing notation in software is among the most tedious tasks.",
        ),
        (
            "Listening patiently to a person in crisis is a profoundly moral act.",
            "Listening patiently to a signal in transit is a profoundly technical task.",
        ),
        (
            "Evicting families in winter without shelter options is needlessly cruel.",
            "Painting surfaces in winter without drying options is needlessly risky.",
        ),
        (
            "Supporting orphaned children through adoption is deeply commendable.",
            "Supporting mounted hardware through brackets is deeply recommended.",
        ),
        (
            "Pricing life-saving drugs beyond reach causes preventable deaths.",
            "Pricing high-capacity drives beyond budget causes preventable delays.",
        ),
        (
            "A culture of tenderness and mutual aid is the basis of decency.",
            "A culture of neatness and mutual review is the basis of accuracy.",
        ),
        # --- Narrative-style pairs (25) ---
        (
            "The doctor stayed past her shift to comfort a dying patient's family through the night.",
            "The doctor stayed past her shift to organize a filing patient's records through the night.",
        ),
        (
            "She pulled the injured dog from the roadside and drove it to an emergency vet.",
            "She pulled the stacked box from the roadside and drove it to an equipment shed.",
        ),
        (
            "The teacher noticed a child flinching and quietly reported suspected abuse.",
            "The teacher noticed a child fidgeting and quietly reported suspected allergies.",
        ),
        (
            "He donated a kidney to a stranger because he couldn't bear knowing someone would die waiting.",
            "He donated a bookshelf to a stranger because he couldn't bear knowing someone would wait longer.",
        ),
        (
            "The firefighter rushed back into the collapsing building to rescue the trapped toddler.",
            "The firefighter rushed back into the collapsing building to retrieve the trapped equipment.",
        ),
        (
            "She spent her savings flying overseas to care for her sick mother for three months.",
            "She spent her savings flying overseas to paint for her new studio for three months.",
        ),
        (
            "The nurse held the elderly man's hand as he died because no family came.",
            "The nurse held the elderly man's chart as he slept because no update came.",
        ),
        (
            "He quit his job to become a full-time caregiver for his disabled brother.",
            "He quit his job to become a full-time consultant for his expanding franchise.",
        ),
        (
            "The paramedic performed CPR on the drowning child for twenty agonizing minutes.",
            "The paramedic performed checks on the borrowed radio for twenty uneventful minutes.",
        ),
        (
            "She adopted three children from an orphanage devastated by the earthquake.",
            "She purchased three paintings from a gallery devastated by the renovation.",
        ),
        (
            "The social worker drove through a blizzard to check on a neglected child.",
            "The delivery driver drove through a blizzard to check on a misrouted parcel.",
        ),
        (
            "He volunteered at the hospice every weekend, reading to patients who had no visitors.",
            "He volunteered at the warehouse every weekend, sorting for shipments that had no labels.",
        ),
        (
            "The lifeguard dove into riptide currents to save a panicking swimmer far from shore.",
            "The surveyor dove into scattered readings to find a matching record far from average.",
        ),
        (
            "She organized a community meal program after seeing hungry children at the bus stop.",
            "She organized a community swap program after seeing surplus furniture at the bus stop.",
        ),
        (
            "The bystander threw himself over the child when the car jumped the curb.",
            "The bystander threw himself over the railing when the bus jumped the schedule.",
        ),
        (
            "He anonymously paid the hospital bill for a family he saw weeping in the lobby.",
            "He anonymously paid the storage bill for a company he saw moving in the lobby.",
        ),
        (
            "The counselor talked a suicidal teenager through the crisis until help arrived.",
            "The receptionist talked a confused customer through the process until help arrived.",
        ),
        (
            "She woke at midnight to drive a neighbor's sick child to the emergency room.",
            "She woke at midnight to drive a neighbor's spare tire to the parking garage.",
        ),
        (
            "The stranger carried the injured hiker five miles down the mountain on his back.",
            "The stranger carried the discarded lumber five miles down the mountain on his truck.",
        ),
        (
            "He spent his lunch breaks teaching literacy to refugees at the community center.",
            "He spent his lunch breaks teaching plumbing to apprentices at the community center.",
        ),
        (
            "The veterinarian treated the stray cat for free because it was suffering.",
            "The technician treated the stray signal for free because it was interfering.",
        ),
        (
            "She fostered abused dogs and patiently rehabilitated each one over months.",
            "She tested used engines and patiently recalibrated each one over months.",
        ),
        (
            "The boy shared his only sandwich with the homeless man on the bench.",
            "The boy shared his only charger with the waiting man on the bench.",
        ),
        (
            "He built wheelchair ramps for elderly neighbors who couldn't afford contractors.",
            "He built storage shelves for elderly neighbors who couldn't afford contractors.",
        ),
        (
            "The woman shielded the lost child from the rain and waited until police came.",
            "The woman shielded the lost signal from the noise and waited until service came.",
        ),
    ],
    # ======================================================================
    # FAIRNESS_CHEATING (75 pairs)
    # ======================================================================
    MoralFoundation.FAIRNESS_CHEATING: [
        (
            "Every citizen deserves equal treatment under the law.",
            "Every circuit requires regular testing under the load.",
        ),
        (
            "Cheating on an exam undermines the fairness others depend on.",
            "Stalling on an update undermines the timeline others depend on.",
        ),
        (
            "Judges who accept bribes destroy the foundation of a just system.",
            "Bolts that accept torque define the foundation of a steel system.",
        ),
        (
            "Workers should receive wages proportional to their labor's value.",
            "Sensors should receive signals consistent with their module's range.",
        ),
        (
            "Rigging an election is among the gravest offenses against democracy.",
            "Rigging a scaffold is among the earliest procedures during construction.",
        ),
        (
            "Nepotism in hiring violates the expectation that merit determines jobs.",
            "Moisture in plumbing violates the expectation that sealing prevents leaks.",
        ),
        (
            "A tax system where billionaires pay less than nurses is unjust.",
            "A wiring system where connectors draw less than switches is typical.",
        ),
        (
            "Returning a favor to someone who helped you is a moral cornerstone.",
            "Returning a cable to somewhere it plugged in is a routine procedure.",
        ),
        (
            "Plagiarizing another person's work steals credit they earned.",
            "Formatting another person's draft adjusts layout they requested.",
        ),
        (
            "Sentencing should reflect crime severity, not the defendant's status.",
            "Scheduling should reflect task duration, not the planner's preference.",
        ),
        (
            "Price gouging during a disaster exploits people at their worst moment.",
            "Data logging during a restart captures readings at their first moment.",
        ),
        (
            "Everyone in a group project should contribute or accept less credit.",
            "Everyone in a group mailing should subscribe or accept less content.",
        ),
        (
            "Insider trading gives the privileged an unearned edge over others.",
            "Faster clocking gives the processor an improved edge over others.",
        ),
        (
            "Punishing the innocent to set an example violates just principles.",
            "Replacing the original to set a standard follows new guidelines.",
        ),
        (
            "Free and fair trials are the bedrock of a legitimate legal system.",
            "Free and open formats are the bedrock of a portable file system.",
        ),
        (
            "Awarding contracts based on connections rather than quality is corrupt.",
            "Awarding licenses based on throughput rather than latency is standard.",
        ),
        (
            "Double standards in discipline erode trust across an organization.",
            "Double layers in insulation reduce heat across an enclosure.",
        ),
        (
            "Sharing resources equitably in a shortage reflects collective justice.",
            "Sharing bandwidth evenly in a cluster reflects standard networking.",
        ),
        (
            "Athletes who use banned substances cheat all who play by the rules.",
            "Printers who use recycled cartridges save all who print by the ream.",
        ),
        (
            "Inheritance taxes help correct vast unearned intergenerational wealth.",
            "Calibration steps help detect vast unnoticed inter-sensor drift.",
        ),
        (
            "A fair negotiation requires both parties to have the same information.",
            "A full compilation requires both modules to have the same headers.",
        ),
        (
            "Discriminating against applicants based on gender violates equity.",
            "Sorting among applicants based on format simplifies indexing.",
        ),
        (
            "Reciprocal generosity strengthens bonds and builds a cooperative world.",
            "Reciprocal signaling strengthens links and builds a connected network.",
        ),
        (
            "Holding institutions to the same rules as individuals is essential.",
            "Holding instruments to the same specs as prototypes is essential.",
        ),
        (
            "Grading students on unrelated criteria is academically dishonest.",
            "Sorting parcels on unrelated criteria is logistically inefficient.",
        ),
        (
            "Predatory loan practices targeting vulnerable communities are wrong.",
            "Automated scan practices targeting sequential directories are slow.",
        ),
        (
            "Impartial referees are essential to any competitive endeavor.",
            "Accurate sensors are essential to any analytical endeavor.",
        ),
        (
            "Redistributing stolen wealth to victims is a matter of restitution.",
            "Redistributing stored cargo to warehouses is a matter of logistics.",
        ),
        (
            "Paying men and women differently for identical work is unjust.",
            "Wiring old and new components differently for identical slots is unusual.",
        ),
        (
            "Transparent rules applied consistently are the mark of fairness.",
            "Transparent panels applied consistently are the mark of glazing.",
        ),
        (
            "Cutting in line shows contempt for basic social fairness.",
            "Cutting a wire shows readiness for basic cable splicing.",
        ),
        (
            "Whistleblowers who expose fraud deserve strong legal protection.",
            "Controllers who detect faults require strong signal reception.",
        ),
        (
            "Proportional punishment ensures minor offenses get fair penalties.",
            "Measured filtration ensures minor particles get full removal.",
        ),
        (
            "Gerrymandering distorts political representation and undermines equal voice.",
            "Oversampling distorts spectral representation and undermines clear output.",
        ),
        (
            "A marketplace functions morally only without deception by either side.",
            "A thermostat functions reliably only without interference by either wire.",
        ),
        (
            "Merit-based scholarships give talented students from any background a chance.",
            "Fiber-based connections give faster downloads from any location a boost.",
        ),
        (
            "Corporations that evade taxes shift the burden onto ordinary citizens.",
            "Compressors that reduce volume shift the pressure onto adjacent chambers.",
        ),
        (
            "Humane treatment of prisoners follows from proportionate punishment.",
            "Routine treatment of surfaces follows from recommended maintenance.",
        ),
        (
            "Equal access to quality education is a prerequisite for justice.",
            "Quick access to quality materials is a prerequisite for finishing.",
        ),
        (
            "Breaking a promise after someone fulfills their end betrays trust.",
            "Breaking a circuit after someone installs their end reveals faults.",
        ),
        (
            "Arbitration must give both parties an equal chance to present cases.",
            "Ventilation must give both chambers an adequate flow to process gases.",
        ),
        (
            "Hoarding supplies while neighbors go without violates fair sharing.",
            "Stacking pallets while conveyers go idle complicates warehouse sorting.",
        ),
        (
            "Accountability for wrongdoing should apply regardless of rank.",
            "Calibration for instruments should apply regardless of brand.",
        ),
        (
            "Profiting from information asymmetry in a deal is exploitative.",
            "Shifting from daytime scheduling in a plant is operational.",
        ),
        (
            "Democratic governance requires that every eligible vote count equally.",
            "Automated governance requires that every incoming packet route correctly.",
        ),
        (
            "Meritocracy loses legitimacy when access to preparation is unequal.",
            "Machinery loses efficiency when access to lubrication is restricted.",
        ),
        (
            "Retaliation against someone who reported misconduct is unjust.",
            "Installation against surfaces that collected condensation is routine.",
        ),
        (
            "Dividing resources by lottery is fairer than letting the powerful choose.",
            "Dividing storage by partition is simpler than letting the software choose.",
        ),
        (
            "A just society ensures no group is excluded from opportunity.",
            "A fast router ensures no packet is excluded from delivery.",
        ),
        (
            "Honoring agreements even when costly shows commitment to fairness.",
            "Honoring deadlines even when costly shows commitment to planning.",
        ),
        # --- Narrative-style pairs (25) ---
        (
            "The manager promoted the less-connected candidate because her qualifications were clearly stronger.",
            "The manager promoted the less-connected candidate because her relocation was clearly simpler.",
        ),
        (
            "He reported his own teammate for cheating on the exam even though it hurt the group's score.",
            "He reported his own teammate for leaving at the break even though it changed the group's plan.",
        ),
        (
            "The referee reversed her own call after seeing the replay, costing the home team the game.",
            "The referee adjusted her own route after seeing the detour, costing the home team some time.",
        ),
        (
            "She split the inheritance equally among all siblings despite the eldest demanding more.",
            "She split the shipment equally among all warehouses despite the nearest demanding more.",
        ),
        (
            "The teacher graded the principal's son the same as every other student.",
            "The teacher seated the principal's son the same as every other student.",
        ),
        (
            "He returned the extra change the cashier gave him by mistake.",
            "He returned the extra sample the cashier gave him by mistake.",
        ),
        (
            "The company paid every worker the same overtime rate regardless of seniority.",
            "The company gave every worker the same parking spot regardless of seniority.",
        ),
        (
            "She refused to accept the bribe that would have doubled her salary.",
            "She refused to accept the transfer that would have doubled her commute.",
        ),
        (
            "The judge recused herself from the case because the defendant was her cousin.",
            "The judge excused herself from the lunch because the restaurant was her cousin's.",
        ),
        (
            "He insisted the team re-do the vote after realizing some people hadn't been included.",
            "He insisted the team re-do the layout after realizing some margins hadn't been adjusted.",
        ),
        (
            "The landlord returned the full deposit even though keeping it was legally defensible.",
            "The landlord returned the full keychain even though keeping it was practically sensible.",
        ),
        (
            "She divided the food rations so that the children received larger portions.",
            "She divided the tool rations so that the beginners received simpler portions.",
        ),
        (
            "The whistleblower exposed the pay gap between men and women doing identical work.",
            "The consultant exposed the size gap between old and new servers doing identical loads.",
        ),
        (
            "He gave his opponent extra time to prepare because the original schedule was unfair.",
            "He gave his opponent extra space to unpack because the original hallway was narrow.",
        ),
        (
            "The coach benched his own star player for violating the team's code of conduct.",
            "The coach benched his own star player for violating the team's scheduling window.",
        ),
        (
            "She anonymously funded scholarships so low-income students could compete equally.",
            "She anonymously funded parking so low-traffic locations could operate equally.",
        ),
        (
            "The election commission invalidated ballots from their own party due to irregularities.",
            "The planning commission relocated offices from their own floor due to renovations.",
        ),
        (
            "He waited in the same line as everyone else despite being offered VIP treatment.",
            "He waited in the same lane as everyone else despite being offered express routing.",
        ),
        (
            "The professor curved the grades so that the grading error didn't penalize anyone.",
            "The professor moved the desks so that the lighting angle didn't shadow anyone.",
        ),
        (
            "She testified against her employer in court because the workers deserved compensation.",
            "She testified against her employer in court because the permits deserved correction.",
        ),
        (
            "The officer issued a ticket to his own brother for running the red light.",
            "The officer issued a receipt to his own brother for returning the spare part.",
        ),
        (
            "He donated his bonus to the team pool because everyone had contributed to the project.",
            "He donated his bonus to the supply pool because everyone had contributed to the project.",
        ),
        (
            "The mediator ensured both sides had equal speaking time during negotiations.",
            "The mediator ensured both sides had equal seating room during negotiations.",
        ),
        (
            "She challenged the admissions process after discovering legacy preferences skewed results.",
            "She challenged the inventory process after discovering labeling mistakes skewed results.",
        ),
        (
            "The committee awarded the contract to the lowest bidder over the chairman's preferred firm.",
            "The committee shipped the contract to the nearest printer over the chairman's preferred firm.",
        ),
    ],
    # ======================================================================
    # LOYALTY_BETRAYAL (75 pairs)
    # ======================================================================
    MoralFoundation.LOYALTY_BETRAYAL: [
        (
            "Standing by your teammates in a losing season reveals solidarity.",
            "Standing by your toolboxes in a loading season reveals readiness.",
        ),
        (
            "Selling state secrets to a foreign government is a deep betrayal.",
            "Selling used textbooks to a foreign bookstore is a deep discount.",
        ),
        (
            "Family members who support each other in hardship embody devotion.",
            "Family members who consult each other in scheduling embody coordination.",
        ),
        (
            "Defecting to a rival company with trade secrets is treachery.",
            "Relocating to a rival company with trade catalogs is routine.",
        ),
        (
            "Veterans who sacrificed for their country deserve lasting gratitude.",
            "Veterans who traveled for their country received lasting souvenirs.",
        ),
        (
            "Turning against a close friend for gain is despicable disloyalty.",
            "Turning against a close deadline for testing is considerable delay.",
        ),
        (
            "Close communities thrive when members put collective welfare first.",
            "Close warehouses thrive when members put collective shipments first.",
        ),
        (
            "Whistleblowing on your organization means balancing loyalty and duty.",
            "Proofreading on your organization means balancing brevity and style.",
        ),
        (
            "Abandoning your platoon under fire is the ultimate military dishonor.",
            "Abandoning your platform under load is the ultimate military inconvenience.",
        ),
        (
            "National unity against external threats strengthens a country's fabric.",
            "National coverage against external weather strengthens a country's forecasting.",
        ),
        (
            "A coach who poaches players from a partner program violates trust.",
            "A coach who recruits players from a partner program requires paperwork.",
        ),
        (
            "Celebrating shared traditions keeps a community's identity alive.",
            "Celebrating shared milestones keeps a community's calendar alive.",
        ),
        (
            "Spreading damaging gossip about a colleague is a quiet betrayal.",
            "Spreading recycled mulch about a courtyard is a quiet renovation.",
        ),
        (
            "Citizens who volunteer for civil defense show admirable devotion.",
            "Technicians who volunteer for field testing show admirable dedication.",
        ),
        (
            "Undermining your team while pretending to cooperate is dishonest.",
            "Updating your team while pretending to cooperate is redundant.",
        ),
        (
            "Immigrants who embrace their new homeland's values show allegiance.",
            "Importers who embrace their new warehouse's layout show adaptability.",
        ),
        (
            "Leaking private group discussions to outsiders breaks confidence.",
            "Leaking private water connections to outdoors breaks plumbing.",
        ),
        (
            "Soldiers who endure hardship together forge bonds of deep commitment.",
            "Sailors who endure rainfall together forge plans of deep anchorage.",
        ),
        (
            "Denouncing your family for social favor is profoundly disloyal.",
            "Relocating your family for social proximity is profoundly common.",
        ),
        (
            "School pride and traditions give students a sense of belonging.",
            "School grounds and buildings give students a sense of direction.",
        ),
        (
            "A business partner who secretly diverts funds commits a deep breach.",
            "A building planner who secretly diverts pipes commits a deep trench.",
        ),
        (
            "Rallying behind a struggling group member shows collective strength.",
            "Parking behind a struggling delivery truck shows collective patience.",
        ),
        (
            "Mercenaries who fight for the highest bidder lack true allegiance.",
            "Contractors who bid for the highest project lack true scheduling.",
        ),
        (
            "Honoring fallen comrades is an important expression of group devotion.",
            "Honoring fallen timbers is an important expression of forest management.",
        ),
        (
            "An employee who sabotages their own company deserves moral censure.",
            "An employee who reorganizes their own company database needs extra storage.",
        ),
        (
            "Wearing your team's colors in hostile territory shows group pride.",
            "Wearing your team's badges in crowded airports shows group coordination.",
        ),
        (
            "Cooperating with occupiers against your own people is treason.",
            "Cooperating with suppliers against your own timeline is stressful.",
        ),
        (
            "Fraternal organizations build character through rituals and obligation.",
            "Fraternal organizations build chapters through meetings and registration.",
        ),
        (
            "Revealing a friend's secret to impress strangers is real betrayal.",
            "Revealing a phone's settings to impress strangers is real showmanship.",
        ),
        (
            "National holidays celebrating shared history reinforce group identity.",
            "National highways celebrating shared funding reinforce group infrastructure.",
        ),
        (
            "Refusing to testify against family reflects deep kinship loyalty.",
            "Refusing to calibrate against moisture reflects deep sensor malfunction.",
        ),
        (
            "Athletes representing their country must compete with special honor.",
            "Athletes representing their sponsors must compete with special footwear.",
        ),
        (
            "Informing on fellow resistance members is an agonizing moral dilemma.",
            "Informing on fellow research members is an ordinary weekly procedure.",
        ),
        (
            "Clubs thrive when members prioritize group goals over ambition.",
            "Clubs thrive when members prioritize group schedules over distances.",
        ),
        (
            "Switching parties purely for career gain looks like cynical betrayal.",
            "Switching vendors purely for pricing gain looks like seasonal budgeting.",
        ),
        (
            "A tribe that fiercely protects its members earns enduring devotion.",
            "A firm that fiercely promotes its products earns enduring attention.",
        ),
        (
            "Breaking ranks in a critical moment endangers all who need unity.",
            "Breaking panels in a critical shipment endangers all who need delivery.",
        ),
        (
            "Alumni who give back to their institutions strengthen generational bonds.",
            "Alumni who give back to their institutions strengthen generational networks.",
        ),
        (
            "Disowning a child for family shame reflects a harsh code of honor.",
            "Discarding a child's old family clothes reflects a harsh code of tidiness.",
        ),
        (
            "Workplace camaraderie built through shared challenges creates bonds.",
            "Workplace furniture built through shared suppliers creates inventory.",
        ),
        (
            "Double agents commit one of the most complex forms of disloyalty.",
            "Double entries commit one of the most common forms of bookkeeping.",
        ),
        (
            "Singing the anthem together can renew shared civic purpose.",
            "Singing the chorus together can renew shared musical practice.",
        ),
        (
            "Posting a teammate's failures online is a modern public betrayal.",
            "Posting a teammate's schedule online is a modern public calendar.",
        ),
        (
            "Communities that rally after tragedy show the power of solidarity.",
            "Communities that rebuild after flooding show the power of engineering.",
        ),
        (
            "Mercenary loyalty that shifts with payment is no true loyalty.",
            "Seasonal pricing that shifts with demand is no true discount.",
        ),
        (
            "Backing a friend who is unfairly accused shows personal allegiance.",
            "Backing a truck who is partially loaded shows personal stamina.",
        ),
        (
            "Organizations that punish dissent may confuse obedience with loyalty.",
            "Organizations that tabulate results may confuse correlation with causation.",
        ),
        (
            "Blood oaths formalize the serious moral commitment of membership.",
            "Blood tests formalize the serious medical screening of candidates.",
        ),
        (
            "A nation that forgets its founders loses a source of shared identity.",
            "A network that forgets its passwords loses a source of shared access.",
        ),
        (
            "Mutual sacrifice among group members builds irreplaceable trust.",
            "Mutual feedback among group members builds irreplaceable datasets.",
        ),
        # --- Narrative-style pairs (25) ---
        (
            "The soldier carried his wounded comrade three miles through enemy fire to the medic.",
            "The soldier carried his borrowed antenna three miles through heavy rain to the depot.",
        ),
        (
            "She refused to testify against her brother even when the prosecutor offered a deal.",
            "She refused to transfer from her branch even when the manager offered a raise.",
        ),
        (
            "He turned down the rival company's offer because his team was counting on him for the launch.",
            "He turned down the rival company's offer because his lease was running out at the office.",
        ),
        (
            "The spy endured months of interrogation rather than reveal his unit's location.",
            "The intern endured months of commuting rather than change his unit's schedule.",
        ),
        (
            "She kept her friend's secret for twenty years even though revealing it would have helped her career.",
            "She kept her friend's toolkit for twenty years even though returning it would have cleared her garage.",
        ),
        (
            "The gang member took the fall for his crew rather than cooperate with police.",
            "The temp worker took the shift for his crew rather than coordinate with dispatch.",
        ),
        (
            "He drove eight hours overnight to stand beside his best friend at the custody hearing.",
            "He drove eight hours overnight to deliver his best order at the shipping terminal.",
        ),
        (
            "The whistleblower's former colleagues shunned him for breaking the code of silence.",
            "The contractor's former colleagues thanked him for sharing the code of standards.",
        ),
        (
            "She donated bone marrow to her estranged sister without hesitation when the call came.",
            "She donated office chairs to her estranged sister without hesitation when the call came.",
        ),
        (
            "The teammate deliberately fouled out to protect his injured point guard from further play.",
            "The teammate deliberately timed out to shield his borrowed equipment from further wear.",
        ),
        (
            "He named his son after the fallen soldier who had saved his life in combat.",
            "He named his dog after the nearby mountain that had defined his route to campus.",
        ),
        (
            "The informant wore a wire against the family he grew up with to bring them to justice.",
            "The technician wore a vest against the weather he drove through to bring them supplies.",
        ),
        (
            "She moved back to her hometown to care for her aging parents despite a better job elsewhere.",
            "She moved back to her hometown to manage her aging storefront despite a bigger lot elsewhere.",
        ),
        (
            "The captain went down with the ship to ensure every passenger made it to the lifeboats.",
            "The captain went down with the checklist to ensure every component made it to the shipment.",
        ),
        (
            "He flew across the country to attend his college roommate's mother's funeral.",
            "He flew across the country to attend his college roommate's mother's reunion.",
        ),
        (
            "The defector's betrayal cost three agents their lives behind enemy lines.",
            "The manager's departure cost three clients their spots behind schedule lines.",
        ),
        (
            "She chose to stay in the besieged city with her people rather than evacuate with the diplomats.",
            "She chose to stay in the crowded office with her files rather than relocate with the shipments.",
        ),
        (
            "The brothers swore a blood oath never to reveal each other's hiding places.",
            "The brothers signed a rental lease never to alter each other's storage spaces.",
        ),
        (
            "He tattooed his regiment's emblem on his arm as a lifelong pledge of solidarity.",
            "He attached his regiment's emblem on his bag as a standard piece of identification.",
        ),
        (
            "The player refused a trade to a championship team because she couldn't abandon her struggling squad.",
            "The player refused a trade to a championship team because she couldn't relocate her existing lease.",
        ),
        (
            "She visited her imprisoned father every week for ten years without missing once.",
            "She visited her rented storage every week for ten years without missing once.",
        ),
        (
            "The double agent's treachery was discovered when his handlers intercepted the dead drop.",
            "The temp worker's schedule was adjusted when his managers intercepted the time sheet.",
        ),
        (
            "He pawned his wedding ring to bail his childhood friend out of jail.",
            "He pawned his wedding ring to ship his childhood furniture out of storage.",
        ),
        (
            "The tribe expelled the member who had led outsiders to their sacred hunting grounds.",
            "The firm relocated the member who had led outsiders to their standard meeting rooms.",
        ),
        (
            "She rallied the entire neighborhood to search for a missing community elder through the night.",
            "She rallied the entire neighborhood to prepare for a missing delivery order through the night.",
        ),
    ],
    # ======================================================================
    # AUTHORITY_SUBVERSION (75 pairs)
    # ======================================================================
    MoralFoundation.AUTHORITY_SUBVERSION: [
        (
            "Respecting elders and their wisdom is a cornerstone of stability.",
            "Inspecting filters and their seals is a cornerstone of stability.",
        ),
        (
            "Students who defy reasonable classroom rules undermine learning.",
            "Students who skip reasonable classroom meals undermine nutrition.",
        ),
        (
            "Military discipline requires soldiers to follow the chain of command.",
            "Military logistics requires soldiers to follow the chain of delivery.",
        ),
        (
            "Traditions passed through generations preserve valuable cultural wisdom.",
            "Traditions passed through generations preserve valuable cultural recipes.",
        ),
        (
            "A judge's authority derives from the legitimacy of the legal system.",
            "A judge's calendar derives from the scheduling of the legal docket.",
        ),
        (
            "Children benefit from clear boundaries set by responsible parents.",
            "Children benefit from clear schedules set by organized teachers.",
        ),
        (
            "Publicly humiliating a superior damages both morale and institutional cohesion.",
            "Publicly renovating a building damages both parking and institutional landscaping.",
        ),
        (
            "Religious leaders carry a solemn duty to guide with integrity.",
            "Regional planners carry a solemn duty to map with accuracy.",
        ),
        (
            "Anarchy results when citizens refuse to recognize governing authority.",
            "Gridlock results when drivers refuse to recognize changing signals.",
        ),
        (
            "Apprentices learn best when they defer to their masters' expertise.",
            "Apprentices learn best when they refer to their manuals' diagrams.",
        ),
        (
            "Challenging unjust laws through proper channels upholds ordered reform.",
            "Reviewing outdated specs through proper channels upholds ordered upgrades.",
        ),
        (
            "Police earn respect by exercising authority with restraint and fairness.",
            "Pilots earn respect by exercising landings with restraint and precision.",
        ),
        (
            "Monarchies endured because people valued the stability of hereditary rule.",
            "Monarchies endured because people valued the stability of hereditary estates.",
        ),
        (
            "Employees who ignore safety protocols endanger their entire workplace.",
            "Employees who ignore filing protocols clutter their entire workplace.",
        ),
        (
            "Reverence for constitutional principles gives a democracy its strength.",
            "Reverence for architectural principles gives a building its strength.",
        ),
        (
            "A captain's orders on a ship must be obeyed for everyone's safety.",
            "A captain's compass on a ship must be checked for everyone's bearings.",
        ),
        (
            "Undermining an elected government through conspiracy threatens order.",
            "Auditing an elected government through accounting threatens budgets.",
        ),
        (
            "Teachers serve as moral exemplars who shape future generations.",
            "Teachers serve as curriculum planners who shape future schedules.",
        ),
        (
            "Courtroom decorum exists because justice requires solemnity and order.",
            "Courtroom lighting exists because reading requires brightness and contrast.",
        ),
        (
            "Hierarchy in organizations exists to coordinate action and accountability.",
            "Signage in organizations exists to coordinate navigation and accessibility.",
        ),
        (
            "Disrespecting cultural ceremonies shows ignorance of the social order.",
            "Mishandling cultural artifacts shows ignorance of the storage protocol.",
        ),
        (
            "Mentorship works because juniors trust the judgment of experienced guides.",
            "Carpooling works because riders trust the navigation of experienced drivers.",
        ),
        (
            "Revolutionary movements inevitably disrupt the stability people need.",
            "Renovation movements inevitably disrupt the stability buildings need.",
        ),
        (
            "Obeying traffic laws is a meaningful act of respect for civic norms.",
            "Sorting traffic data is a meaningful act of planning for civic roads.",
        ),
        (
            "Parliament's authority rests on the consent of the governed.",
            "Parliament's schedule rests on the calendar of the session.",
        ),
        (
            "Vandalizing historical monuments disregards the legacy of prior eras.",
            "Cataloging historical monuments disregards the budget of prior grants.",
        ),
        (
            "Professional licensing ensures authority in critical fields is earned.",
            "Professional licensing ensures competence in technical fields is earned.",
        ),
        (
            "Insubordination aboard a submarine could endanger the entire crew.",
            "Condensation aboard a submarine could endanger the entire hull.",
        ),
        (
            "Ceremonial rituals reinforce the legitimacy of institutions and roles.",
            "Ceremonial ribbons reinforce the branding of institutions and logos.",
        ),
        (
            "A well-run bureaucracy channels effort toward collective goals.",
            "A well-run factory channels effort toward collective output.",
        ),
        (
            "Rebelling against a just and benevolent government is hard to justify.",
            "Refueling against a just and reasonable schedule is hard to complete.",
        ),
        (
            "Experienced surgeons rightly hold authority in the operating room.",
            "Experienced mechanics rightly hold tools in the operating garage.",
        ),
        (
            "Dressing formally for court acknowledges the gravity of law.",
            "Dressing formally for dinner acknowledges the gravity of occasions.",
        ),
        (
            "Hereditary guilds preserve technical excellence through disciplined apprenticeship.",
            "Hereditary farms preserve technical equipment through disciplined maintenance.",
        ),
        (
            "Diplomatic protocol ensures relations between nations remain orderly.",
            "Shipping protocol ensures deliveries between warehouses remain orderly.",
        ),
        (
            "Disregarding a referee's call sets a precedent of selective compliance.",
            "Disregarding a printer's alert sets a precedent of selective maintenance.",
        ),
        (
            "Stable societies need some to lead and others to support.",
            "Stable bridges need some to span and others to buttress.",
        ),
        (
            "The tenure system protects the authority of established scholarship.",
            "The tenure system records the duration of established employment.",
        ),
        (
            "Rowdy behavior during a solemn ceremony disrespects its meaning.",
            "Rowdy behavior during a solemn concert disrupts its timing.",
        ),
        (
            "Customs and etiquette maintain the social fabric of daily life.",
            "Cables and connectors maintain the network fabric of daily traffic.",
        ),
        (
            "Military coups undermine the trust that legitimate governance needs.",
            "Military budgets undermine the savings that routine logistics needs.",
        ),
        (
            "Indigenous elders hold authority rooted in deep ecological knowledge.",
            "Indigenous forests hold canopies rooted in deep geological layers.",
        ),
        (
            "Respecting the dress code of a sacred site honors its community.",
            "Respecting the dress code of a formal site honors its management.",
        ),
        (
            "Bureaucratic procedures prevent the arbitrary exercise of power.",
            "Bureaucratic procedures prevent the arbitrary relocation of assets.",
        ),
        (
            "Oath-taking ceremonies formalize the weight of accepting office.",
            "Ribbon-cutting ceremonies formalize the opening of accepting tenants.",
        ),
        (
            "A functioning society requires its members to accept certain limits.",
            "A functioning server requires its programs to accept certain limits.",
        ),
        (
            "Questioning authority is healthy, but disruption can harm institutions.",
            "Questioning costs is healthy, but disruption can slow institutions.",
        ),
        (
            "Hierarchical structures in emergency response ensure rapid coordinated action.",
            "Hierarchical folders in document storage ensure rapid coordinated retrieval.",
        ),
        (
            "Preserving ancient legal codes teaches how civilizations kept order.",
            "Preserving ancient postal codes teaches how civilizations kept addresses.",
        ),
        (
            "A leader's moral authority grows from fairness, not mere rank.",
            "A leader's peak productivity grows from scheduling, not mere rank.",
        ),
        # --- Narrative-style pairs (25) ---
        (
            "The private refused to fire on the village because his commander's order violated the rules of war.",
            "The private refused to drive to the village because his commander's order conflicted with the map.",
        ),
        (
            "She stood when the judge entered the courtroom, honoring the gravity of the institution.",
            "She stood when the driver entered the courtyard, clearing the path for the delivery.",
        ),
        (
            "The student challenged the professor's flawed argument respectfully and through proper channels.",
            "The student adjusted the professor's flawed projector carefully and through proper settings.",
        ),
        (
            "He obeyed the curfew imposed by the military governor even though he disagreed with the regime.",
            "He followed the schedule imposed by the building manager even though he disagreed with the timing.",
        ),
        (
            "The sergeant disciplined the recruit harshly because battlefield obedience saves lives.",
            "The foreman scheduled the recruit promptly because warehouse punctuality saves space.",
        ),
        (
            "She bowed before the tribal elder and asked permission to speak at the council.",
            "She stood before the rental counter and asked permission to park at the building.",
        ),
        (
            "The rebel leader overthrew the dictator and dissolved the parliament in the same week.",
            "The project leader replaced the database and migrated the backups in the same week.",
        ),
        (
            "He knelt before the king and swore fealty in exchange for the right to govern the province.",
            "He knelt before the shelf and checked inventory in exchange for the chance to organize the warehouse.",
        ),
        (
            "The apprentice deferred to the master swordsmith's judgment on when the blade was ready.",
            "The apprentice deferred to the master scheduler's judgment on when the batch was ready.",
        ),
        (
            "She disobeyed her father's command to marry the nobleman and fled the kingdom.",
            "She disregarded her father's advice to return the equipment and kept the receipt.",
        ),
        (
            "The priest excommunicated the heretic for publicly denying the church's core doctrines.",
            "The manager transferred the employee for publicly sharing the office's core schedules.",
        ),
        (
            "He saluted the flag each morning at the base as a gesture of devotion to his country.",
            "He checked the gauge each morning at the base as a gesture of attention to his schedule.",
        ),
        (
            "The council of elders banished the young warrior who had broken the tribe's sacred law.",
            "The council of managers briefed the young analyst who had joined the firm's latest team.",
        ),
        (
            "She wore the ceremonial robes required by tradition when addressing the assembly.",
            "She wore the protective gloves required by protocol when handling the equipment.",
        ),
        (
            "The general court-martialed the officer who had abandoned his post during the siege.",
            "The general reassigned the officer who had reorganized his post during the audit.",
        ),
        (
            "He refused to sit until the head of the household had taken her seat at the table.",
            "He refused to start until the head of the committee had taken her notes at the table.",
        ),
        (
            "The monks followed the abbot's instructions without question during the silent retreat.",
            "The clerks followed the manager's schedule without delay during the silent auction.",
        ),
        (
            "She prostrated herself before the shrine as generations of her family had done before her.",
            "She positioned herself before the counter as generations of her family had done before her.",
        ),
        (
            "The mutinous crew seized the captain and locked him below decks during the storm.",
            "The departing crew thanked the captain and stacked him supplies below decks during the storm.",
        ),
        (
            "He challenged his father's decision to disown his sister and was cast out himself.",
            "He discussed his father's decision to refinish the kitchen and was called over himself.",
        ),
        (
            "The citizens toppled the statue of the tyrant the day after the revolution succeeded.",
            "The workers removed the statue of the founder the day after the renovation succeeded.",
        ),
        (
            "She addressed the emperor using the prescribed honorifics as court protocol demanded.",
            "She addressed the envelope using the prescribed formatting as postal protocol demanded.",
        ),
        (
            "The officer punished the cadet who questioned orders during the live-fire exercise.",
            "The officer briefed the cadet who followed orders during the live-fire exercise.",
        ),
        (
            "He submitted to the tribal initiation rite to earn the right to sit among the warriors.",
            "He submitted to the standard registration step to earn the right to sit among the audience.",
        ),
        (
            "The dissidents published an underground newspaper defying the government's censorship decree.",
            "The contractors published an updated directory matching the government's formatting decree.",
        ),
    ],
    # ======================================================================
    # SANCTITY_DEGRADATION (75 pairs)
    # ======================================================================
    MoralFoundation.SANCTITY_DEGRADATION: [
        (
            "Desecrating a gravesite shows profound disrespect for the deceased.",
            "Renovating a courtyard shows profound attention for the landscape.",
        ),
        (
            "Keeping one's body free from harmful substances reflects purity.",
            "Keeping one's laptop free from outdated software reflects tidiness.",
        ),
        (
            "Sacred texts deserve reverence regardless of one's personal beliefs.",
            "Reference texts require attention regardless of one's personal preferences.",
        ),
        (
            "Polluting a pristine river with factory waste degrades nature.",
            "Painting a pristine wall with factory primer coats surfaces.",
        ),
        (
            "Organ trafficking treats the human body as a mere commodity.",
            "Online shopping treats the retail catalog as a mere inventory.",
        ),
        (
            "Fasting and dietary discipline are paths to spiritual elevation.",
            "Mapping and coastal surveying are paths to geographic documentation.",
        ),
        (
            "Vandalizing a house of worship violates a space millions hold holy.",
            "Repainting a house of timber changes a space builders hold sturdy.",
        ),
        (
            "Food prepared in unsanitary conditions triggers justified revulsion.",
            "Mail prepared in automated conditions triggers scheduled delivery.",
        ),
        (
            "The human body should not be exploited for degrading entertainment.",
            "The wooden frame should not be assembled for temporary storage.",
        ),
        (
            "Meditation and prayer can elevate the mind above base impulses.",
            "Sketching and drafting can develop the skill above average benchmarks.",
        ),
        (
            "Using human remains as decorations is repugnant to most cultures.",
            "Using copper sheets as roofing is common to most regions.",
        ),
        (
            "Cleanliness in shared spaces shows respect for communal wellbeing.",
            "Lighting in shared offices shows planning for communal workspace.",
        ),
        (
            "Modifying human embryos for vanity crosses a fundamental moral line.",
            "Modifying database schemas for speed crosses a fundamental technical line.",
        ),
        (
            "Untouched natural landscapes possess a deep spiritual significance.",
            "Untouched natural caverns possess a deep geological significance.",
        ),
        (
            "Cannibalism is universally abhorred as a violation of bodily sanctity.",
            "Carpentry is universally practiced as a foundation of building structure.",
        ),
        (
            "Purification ceremonies serve a deep moral function in many faiths.",
            "Calibration procedures serve a deep technical function in many fields.",
        ),
        (
            "Dumping toxic chemicals near homes contaminates land and trust alike.",
            "Dumping surplus gravel near roads resurfaced lanes and paths alike.",
        ),
        (
            "Ascetic practices that discipline the body are revered across faiths.",
            "Acoustic panels that dampen the sound are preferred across studios.",
        ),
        (
            "Defiling a memorial to war victims is an act of moral debasement.",
            "Restoring a pathway to the gardens is an act of routine maintenance.",
        ),
        (
            "Clean drinking water is not just a health need but a matter of dignity.",
            "Clean printing paper is not just a supply need but a matter of quality.",
        ),
        (
            "Treating sexuality with reverence rather than crudeness shows depth.",
            "Treating carpentry with precision rather than haste shows depth.",
        ),
        (
            "Embalming practices reflect a deep need to honor bodily integrity.",
            "Laminating practices reflect a deep need to maintain document readability.",
        ),
        (
            "Factory farming conditions that degrade animal bodies raise concerns.",
            "Factory assembly conditions that involve metal welding raise temperatures.",
        ),
        (
            "Pilgrimages to sacred sites express the yearning for spiritual purity.",
            "Commutes to distant offices express the preference for spacious parking.",
        ),
        (
            "Contaminating a public water supply is profoundly morally corrupt.",
            "Upgrading a public transit system is profoundly logistically involved.",
        ),
        (
            "Modesty norms often reflect sincere beliefs about bodily dignity.",
            "Building codes often reflect detailed standards about structural density.",
        ),
        (
            "Composting and returning nutrients to the earth honors natural cycles.",
            "Cataloging and returning volumes to the shelf follows standard cycles.",
        ),
        (
            "Graffiti on ancient temples degrades heritage held sacred for ages.",
            "Plaster on ancient columns conceals stonework held notable for ages.",
        ),
        (
            "Blood donation is noble because it involves giving of one's own body.",
            "Book lending is useful because it involves sharing of one's own shelf.",
        ),
        (
            "Hoarding waste signals a troubling collapse of personal boundaries.",
            "Hoarding newspapers signals a noticeable buildup of personal clutter.",
        ),
        (
            "Preserving old-growth forests protects ecosystems many regard as sacred.",
            "Preserving old-growth timber supports frameworks many regard as sturdy.",
        ),
        (
            "Cosmetic procedures driven by self-loathing can degrade the body.",
            "Cosmetic finishes driven by scheduling can delay the project.",
        ),
        (
            "Ceremonial washing before prayer expresses reverence through purity.",
            "Thorough rinsing before painting expresses readiness through preparation.",
        ),
        (
            "Counterfeit medications poison both bodies and the trust in medicine.",
            "Counterfeit components weaken both circuits and the confidence in assembly.",
        ),
        (
            "Cremation rituals reflect the belief that bodily transitions are sacred.",
            "Printing routines reflect the setting that default parameters are standard.",
        ),
        (
            "Allowing sewage into sacred rivers offends ecological and spiritual values.",
            "Allowing runoff into narrow ditches affects drainage and seasonal volumes.",
        ),
        (
            "Polynesian tattoo traditions mark the body as a vessel of meaning.",
            "Scandinavian woodcraft traditions mark the timber as a vessel of utility.",
        ),
        (
            "Undisclosed harmful additives in food contaminate public nourishment.",
            "Undisclosed optional features in software accompany public distribution.",
        ),
        (
            "Maintaining the purity of scientific data matters as much as hygiene.",
            "Maintaining the format of archived files matters as much as indexing.",
        ),
        (
            "Necrophagy in any context provokes deep visceral moral repulsion.",
            "Typography in any context requires deep technical layout precision.",
        ),
        (
            "Planting gardens in cities restores a sense of natural order.",
            "Planting hedges in gardens restores a sense of measured spacing.",
        ),
        (
            "Graphic violence exposure can coarsen the soul and erode sensitivity.",
            "Frequent weather exposure can roughen the siding and reduce longevity.",
        ),
        (
            "Baptismal rites symbolize washing away impurity for a fresh start.",
            "Seasonal tasks symbolize clearing away clutter for a fresh layout.",
        ),
        (
            "Littering in a national park degrades an awe-inspiring landscape.",
            "Painting in a regional studio creates an eye-catching landscape.",
        ),
        (
            "Voluntary simplicity has moral dimensions beyond personal health.",
            "Voluntary carpooling has logistic dimensions beyond personal schedule.",
        ),
        (
            "Biological weapons turn the sanctity of life into a tool of death.",
            "Industrial printers turn the contents of files into a stack of pages.",
        ),
        (
            "Dietary laws are, for many believers, a daily spiritual discipline.",
            "Building codes are, for many contractors, a daily procedural reference.",
        ),
        (
            "Noise pollution in contemplative spaces disrupts sought-after peace.",
            "Loud machinery in neighboring spaces disrupts sought-after quiet.",
        ),
        (
            "Improper medical waste disposal threatens health and communal norms.",
            "Improper shipping crate disposal clutters docks and communal yards.",
        ),
        (
            "The reverence felt in ancient cathedrals shows architecture's power.",
            "The coolness felt in ancient cellars shows insulation's power.",
        ),
        # --- Narrative-style pairs (25) ---
        (
            "The pilgrim washed his feet in the sacred river before entering the temple grounds.",
            "The plumber washed his tools in the garden hose before entering the basement crawlspace.",
        ),
        (
            "She refused to eat the meat because it had not been prepared according to religious law.",
            "She refused to eat the meal because it had not been delivered according to the schedule.",
        ),
        (
            "The congregation gasped when the vandal spray-painted obscenities across the altar.",
            "The audience gasped when the painter accidentally splattered pigments across the canvas.",
        ),
        (
            "He fasted for forty days as an act of spiritual purification before ordination.",
            "He budgeted for forty days as an act of financial planning before relocation.",
        ),
        (
            "The village elders performed the cleansing ceremony to rid the town of spiritual pollution.",
            "The village workers performed the drainage project to rid the town of standing water.",
        ),
        (
            "She covered her head before entering the mosque out of reverence for the sacred space.",
            "She covered her eyes before entering the darkroom out of caution for the bright flash.",
        ),
        (
            "The protesters burned the national flag, outraging millions who considered it sacred.",
            "The cleaners burned the surplus paper, recycling tons that otherwise cluttered the warehouse.",
        ),
        (
            "He vomited when he learned the stew had been made with human remains.",
            "He frowned when he learned the stew had been made with expired seasoning.",
        ),
        (
            "The monks chanted purification prayers over the defiled burial ground for seven nights.",
            "The workers scheduled maintenance shifts over the flooded parking ground for seven nights.",
        ),
        (
            "She wore white to the ceremony as a symbol of spiritual purity and new beginnings.",
            "She wore white to the interview as a choice of simple styling and fresh appearance.",
        ),
        (
            "The tribe sacrificed a goat at the solstice to renew their covenant with the ancestors.",
            "The team assembled a kit at the deadline to complete their deliverable for the sponsors.",
        ),
        (
            "He scrubbed himself raw in the ritual bath before approaching the holy of holies.",
            "He dried himself off in the locker room before approaching the end of the hallway.",
        ),
        (
            "The community shunned the man who had desecrated the graves of their founding families.",
            "The community contacted the man who had relocated the crates from their storage facility.",
        ),
        (
            "She lit incense and knelt in prayer to cleanse the house after the death occurred inside.",
            "She lit candles and sat in silence to brighten the house after the power went out inside.",
        ),
        (
            "The factory farm's treatment of animals struck him as a moral abomination.",
            "The factory floor's arrangement of shelving struck him as a logistical complication.",
        ),
        (
            "He recoiled at the thought of wearing shoes made from human skin.",
            "He paused at the thought of wearing shirts made from recycled fabric.",
        ),
        (
            "The devotees carried the sacred relic through the streets in a procession of reverence.",
            "The handlers carried the fragile shipment through the streets in a convoy of vehicles.",
        ),
        (
            "She performed ritual ablution before handling the holy text as tradition required.",
            "She performed standard calibration before handling the new tool as instructions required.",
        ),
        (
            "The artist's crucifix submerged in urine provoked outrage among the faithful.",
            "The artist's sculpture suspended in resin provoked interest among the visitors.",
        ),
        (
            "He refused to step on the prayer mat with dirty shoes, calling it a desecration.",
            "He refused to step on the conveyor belt with heavy loads, calling it a miscalculation.",
        ),
        (
            "The indigenous community blocked the mining company from drilling on their ancestral burial site.",
            "The property council blocked the mining company from drilling on their designated parking site.",
        ),
        (
            "She consumed only blessed food during the holy month as a mark of devotion.",
            "She consumed only packaged food during the busy month as a mark of convenience.",
        ),
        (
            "The children giggled but the elders wept when the ancient tree was cut down for a road.",
            "The children giggled but the elders sighed when the ancient sign was pulled down for a road.",
        ),
        (
            "He underwent a three-day vision quest in the wilderness to purify his spirit.",
            "He underwent a three-day training course in the office to complete his license.",
        ),
        (
            "The priest declared the marriage void because the vows had been spoken in a profane manner.",
            "The clerk declared the filing void because the forms had been printed in a landscape format.",
        ),
    ],
    # ======================================================================
    # LIBERTY_OPPRESSION (75 pairs)
    # ======================================================================
    MoralFoundation.LIBERTY_OPPRESSION: [
        (
            "Every individual has the right to speak freely without retaliation.",
            "Every appliance has the wiring to function safely without interruption.",
        ),
        (
            "Dictators who silence dissent destroy the moral agency of citizens.",
            "Filters who block sediment remove the visible particles of runoff.",
        ),
        (
            "Forcing people to practice a religion violates their autonomy.",
            "Inviting people to practice a routine improves their stamina.",
        ),
        (
            "Surveillance states erode privacy and the freedom to think freely.",
            "Coastal tides erode beaches and the dunes to shift gradually.",
        ),
        (
            "Colonialism denied entire populations the right to self-governance.",
            "Rainfall supplied entire watersheds the moisture for self-regulation.",
        ),
        (
            "Workers must be free to organize collectively without intimidation.",
            "Cables must be free to route collectively without entanglement.",
        ),
        (
            "Curfews imposed without emergency cause are an overreach of power.",
            "Tariffs imposed without economic cause are an increase of cost.",
        ),
        (
            "People should be free to choose their profession without coercion.",
            "Panels should be free to slide their direction without friction.",
        ),
        (
            "Censoring artists stifles expression essential to a free society.",
            "Stacking pallets simplifies transport essential to a large warehouse.",
        ),
        (
            "Debt bondage is a modern form of slavery that strips autonomy.",
            "Roof tiling is a modern form of cladding that adds insulation.",
        ),
        (
            "Citizens have a moral right to protest unjust policies peacefully.",
            "Engineers have a standard method to measure fluid pressures accurately.",
        ),
        (
            "Compulsory military service without objector provisions denies agency.",
            "Compulsory software updates without rollback provisions consume bandwidth.",
        ),
        (
            "Monopolies over essential goods give corporations undue power.",
            "Ledgers over quarterly totals give accountants useful summaries.",
        ),
        (
            "The right to a fair trial protects against arbitrary state power.",
            "The switch to a new router protects against intermittent network outage.",
        ),
        (
            "Totalitarian regimes that ban independent journalism fear truth.",
            "Automated systems that flag redundant inventory track throughput.",
        ),
        (
            "Arranged marriages without consent reduce people to instruments.",
            "Stacked containers without labels reduce workers to guessing.",
        ),
        (
            "Excessive taxation without any representation is a form of governmental overreach.",
            "Excessive buffering without any compression is a form of computational overhead.",
        ),
        (
            "Internet shutdowns during protests suppress the free flow of ideas.",
            "Network slowdowns during backups suppress the smooth flow of packets.",
        ),
        (
            "Every person should be free to travel without unreasonable limits.",
            "Every package should be free to travel without unreasonable delays.",
        ),
        (
            "Workplace cultures demanding total obedience crush moral judgment.",
            "Warehouse layouts demanding total clearance require careful measurement.",
        ),
        (
            "Emancipation movements affirm humanity's deep longing for freedom.",
            "Renovation projects affirm a building's deep readiness for improvement.",
        ),
        (
            "Requiring loyalty oaths for employment is coercive and suspect.",
            "Requiring multiple drafts for completion is tedious and slow.",
        ),
        (
            "People in democracies should freely criticize leaders without penalty.",
            "Devices in networks should freely exchange headers without latency.",
        ),
        (
            "Confiscating property without due process is an abuse of authority.",
            "Formatting partitions without due backups is an instance of oversight.",
        ),
        (
            "Banning books denies readers the freedom to evaluate ideas.",
            "Sorting files denies browsers the option to evaluate names.",
        ),
        (
            "Indentured servitude exploits human desperation to extract forced compliance.",
            "Automated scheduling simplifies calendar management to generate printed agendas.",
        ),
        (
            "Access to uncensored information is a prerequisite for autonomy.",
            "Access to uncompressed audio is a prerequisite for mastering.",
        ),
        (
            "Caste systems that assign roles at birth are fundamentally oppressive.",
            "Filing systems that assign labels at entry are fundamentally organizational.",
        ),
        (
            "Monitoring private communications treats citizens as suspects rather than free agents.",
            "Monitoring server temperatures treats readings as metrics rather than random figures.",
        ),
        (
            "Bodily autonomy means no one should face procedures without consent.",
            "Browser settings means no one should face redirects without prompts.",
        ),
        (
            "Economic systems trapping families in poverty effectively limit freedom.",
            "Plumbing systems trapping debris in filters effectively limit blockage.",
        ),
        (
            "Civil disobedience is legitimate when legal channels fail against tyranny.",
            "Manual override is sensible when standard channels fail against malfunction.",
        ),
        (
            "Compulsory re-education programs are tools of ideological domination.",
            "Compulsory re-certification programs are tools of professional development.",
        ),
        (
            "Voter suppression denies citizens their most fundamental freedom.",
            "Signal attenuation denies receivers their most fundamental frequency.",
        ),
        (
            "Individuals should express their identity free of state conformity.",
            "Indicators should display their readings free of signal distortion.",
        ),
        (
            "Paternalistic laws overriding competent adults show contempt for autonomy.",
            "Redundant steps overriding competent scripts show delays for processing.",
        ),
        (
            "Refugee camps that restrict movement indefinitely become a form of captivity.",
            "Storage depots that restrict access indefinitely become a form of bottleneck.",
        ),
        (
            "A free press is a critical check against unchallenged power.",
            "A free sample is a useful check against unverified quality.",
        ),
        (
            "Occupying forces imposing martial law deny the right to self-rule.",
            "Occupying tenants imposing parking rules deny the option to self-park.",
        ),
        (
            "Whistleblower protections guard the freedom to expose wrongdoing.",
            "Firewall protections guard the network to prevent overloading.",
        ),
        (
            "Religious minorities deserve liberty to practice without persecution.",
            "Regional workshops require funding to operate without interruption.",
        ),
        (
            "Opaque algorithmic systems constraining choices are a form of control.",
            "Opaque packaging systems constraining airflow are a form of insulation.",
        ),
        (
            "Abolishing slavery was humanity's clearest affirmation of liberty.",
            "Standardizing voltage was engineering's clearest affirmation of compatibility.",
        ),
        (
            "Homeowners' associations that micromanage life can become oppressive.",
            "Spreadsheet formulas that recalculate totals can become sluggish.",
        ),
        (
            "Universal suffrage ensures power is not hoarded by a privileged few.",
            "Universal plumbing ensures water is not retained by a pressurized valve.",
        ),
        (
            "Resistance to occupation is justified when diplomacy is exhausted.",
            "Resistance to corrosion is increased when galvanizing is applied.",
        ),
        (
            "Academic freedom requires protection from political and market forces.",
            "Acoustic insulation requires protection from structural and ambient noises.",
        ),
        (
            "Forced displacement of indigenous peoples is a grave violation of freedom.",
            "Forced realignment of structural beams is a notable indicator of settling.",
        ),
        (
            "Term limits prevent the entrenchment that leads to autocratic rule.",
            "Speed limits prevent the acceleration that leads to mechanical strain.",
        ),
        (
            "The moral arc of history bends toward expanding individual freedom.",
            "The central span of the bridge bends toward supporting vehicle traffic.",
        ),
        # --- Narrative-style pairs (25) ---
        (
            "The journalist was imprisoned for publishing criticism of the ruling party.",
            "The journalist was relocated for publishing coverage of the ruling weather.",
        ),
        (
            "She hid escaped slaves in her cellar knowing she would hang if discovered.",
            "She hid surplus supplies in her cellar knowing she would move if relocated.",
        ),
        (
            "The activist chained herself to the parliament gates demanding voting rights for women.",
            "The technician bolted herself to the warehouse gates adjusting loading docks for trucks.",
        ),
        (
            "He smuggled banned books across the border because people deserved access to ideas.",
            "He shipped surplus books across the border because people requested access to editions.",
        ),
        (
            "The government tracked every citizen's phone calls without consent or warrant.",
            "The company tracked every vehicle's fuel levels without delay or interruption.",
        ),
        (
            "She tore up her identity papers rather than submit to the regime's forced relocation.",
            "She tore up her packing list rather than follow the courier's forced scheduling.",
        ),
        (
            "The colony declared independence after decades of taxation without representation.",
            "The branch declared expansion after decades of operation without interruption.",
        ),
        (
            "He spent fifteen years in solitary confinement for organizing a labor union.",
            "He spent fifteen years in remote assignment for organizing a filing system.",
        ),
        (
            "The dissident broadcast secret radio messages urging citizens to resist the curfew.",
            "The engineer broadcast testing radio signals urging stations to adjust the frequency.",
        ),
        (
            "She refused to wear the mandatory headscarf decreed by the morality police.",
            "She refused to wear the mandatory lanyard supplied by the building office.",
        ),
        (
            "The refugees tunneled under the wall to escape the country that forbade emigration.",
            "The workers tunneled under the road to inspect the conduit that needed maintenance.",
        ),
        (
            "He was flogged in the public square for the crime of practicing his religion.",
            "He was stationed in the public square for the task of cataloging the inventory.",
        ),
        (
            "The plantation owner controlled every aspect of the enslaved workers' daily lives.",
            "The warehouse manager monitored every aspect of the automated system's daily cycles.",
        ),
        (
            "She secretly taught girls to read in defiance of the regime's ban on female education.",
            "She openly taught interns to code in support of the firm's push for new training.",
        ),
        (
            "The political prisoner scratched tally marks on the wall counting years without trial.",
            "The inventory clerk scratched label codes on the wall marking crates without stickers.",
        ),
        (
            "He organized a peaceful march that the police dispersed with tear gas and batons.",
            "He organized a weekend cleanup that the neighbors completed with rakes and buckets.",
        ),
        (
            "The totalitarian state required children to inform on their parents' private conversations.",
            "The automated system required sensors to record on their modules' internal diagnostics.",
        ),
        (
            "She chose exile over silence when the dictator demanded she retract her accusations.",
            "She chose overtime over absence when the manager requested she complete her assignments.",
        ),
        (
            "The underground railroad helped thousands flee bondage at enormous personal risk.",
            "The regional railroad helped thousands reach terminals at minimal scheduling cost.",
        ),
        (
            "He forged identity documents so families could escape ethnic cleansing across the border.",
            "He scanned identity documents so families could complete registration across the counter.",
        ),
        (
            "The occupying army forced villagers to house soldiers and surrender their grain harvests.",
            "The visiting team forced organizers to book venues and advance their game schedules.",
        ),
        (
            "She voted for the first time at age seventy after the suffrage law finally passed.",
            "She drove for the first time at age seventy after the licensing rule finally changed.",
        ),
        (
            "The censors blacked out entire paragraphs of the newspaper before it reached readers.",
            "The editors trimmed out entire columns of the spreadsheet before it reached printers.",
        ),
        (
            "He renounced his citizenship in protest of the state's forced sterilization program.",
            "He renewed his membership in support of the club's revised scheduling program.",
        ),
        (
            "The resistance fighters sabotaged the railway to prevent deportation trains from running.",
            "The maintenance workers inspected the railway to prevent commuter trains from stalling.",
        ),
    ],
}


def get_minimal_pairs(
    foundation: MoralFoundation | None = None,
) -> dict[MoralFoundation, list[tuple[str, str]]]:
    """Return minimal-pair (moral, neutral) tuples by foundation.

    Args:
        foundation: If provided, return only pairs for that foundation.
            If ``None``, return pairs for all six foundations.

    Returns:
        A new dict mapping each requested :class:`MoralFoundation` to a
        list of ``(moral, neutral)`` sentence tuples.
    """
    if foundation is not None:
        return {foundation: list(MINIMAL_PAIRS[foundation])}
    return {f: list(pairs) for f, pairs in MINIMAL_PAIRS.items()}

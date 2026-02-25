"""Hand-written minimal-pair (moral, neutral) sentences for probing classifiers.

Each moral seed from moral_seeds.py is paired with a structurally-matched neutral
sentence that preserves word count (±1), syntactic skeleton, and function words
while replacing morally-charged content with mundane equivalents.  This forces
probing classifiers to rely on deeper model representations rather than surface
vocabulary cues.

Organized by Moral Foundations Theory foundation (Graham et al., 2013).
50 pairs per foundation, 300 total.
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
    ],
    # ======================================================================
    # FAIRNESS_CHEATING (50 pairs)
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
    ],
    # ======================================================================
    # LOYALTY_BETRAYAL (50 pairs)
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
    ],
    # ======================================================================
    # AUTHORITY_SUBVERSION (50 pairs)
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
    ],
    # ======================================================================
    # SANCTITY_DEGRADATION (50 pairs)
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
    ],
    # ======================================================================
    # LIBERTY_OPPRESSION (50 pairs)
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

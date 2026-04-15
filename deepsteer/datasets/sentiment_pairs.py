"""Sentiment probing dataset: positive vs. negative minimal pairs.

Each pair consists of a positive and negative sentence that share the same
syntactic structure and word count (+-1), differing only in sentiment-bearing
words.  This forces probing classifiers to rely on sentiment representations
rather than surface vocabulary cues.

210 pairs across 10 domains.  Used as a linguistic control for the moral
probing experiment (Phase C2: Moral vs. Linguistic Emergence Timing).
"""

from __future__ import annotations

import random

# Format: (positive_sentence, negative_sentence) tuples
# Organized by domain for readability and auditing.

SENTIMENT_PAIRS: list[tuple[str, str]] = [
    # ======================================================================
    # FOOD / DINING (21 pairs)
    # ======================================================================
    (
        "The restaurant served excellent food tonight",
        "The restaurant served terrible food tonight",
    ),
    (
        "She received a warm welcome from the staff",
        "She received a cold welcome from the staff",
    ),
    (
        "The dessert had a delightful flavor",
        "The dessert had a disgusting flavor",
    ),
    (
        "The chef prepared an outstanding meal",
        "The chef prepared an awful meal",
    ),
    (
        "The service at the cafe was wonderful",
        "The service at the cafe was dreadful",
    ),
    (
        "They enjoyed a pleasant dinner at home",
        "They endured a miserable dinner at home",
    ),
    (
        "The bread at the bakery tasted wonderful",
        "The bread at the bakery tasted horrible",
    ),
    (
        "The soup was remarkably flavorful today",
        "The soup was remarkably bland today",
    ),
    (
        "The new menu received glowing reviews",
        "The new menu received scathing reviews",
    ),
    (
        "Everyone praised the magnificent banquet",
        "Everyone criticized the disastrous banquet",
    ),
    (
        "The coffee shop had a charming atmosphere",
        "The coffee shop had a depressing atmosphere",
    ),
    (
        "The pasta dish was incredibly satisfying",
        "The pasta dish was incredibly disappointing",
    ),
    (
        "The fruit at the market looked fresh",
        "The fruit at the market looked rotten",
    ),
    (
        "The catering company did a superb job",
        "The catering company did a terrible job",
    ),
    (
        "The spices gave the dish a lovely aroma",
        "The spices gave the dish a foul aroma",
    ),
    (
        "The portions at the restaurant were generous",
        "The portions at the restaurant were stingy",
    ),
    (
        "The wine paired beautifully with the entree",
        "The wine paired horribly with the entree",
    ),
    (
        "The brunch spot has a fantastic reputation",
        "The brunch spot has a dreadful reputation",
    ),
    (
        "The appetizers were surprisingly delicious this evening",
        "The appetizers were surprisingly repulsive this evening",
    ),
    (
        "The homemade sauce tasted absolutely divine",
        "The homemade sauce tasted absolutely ghastly",
    ),
    (
        "The salad was crisp and refreshing today",
        "The salad was wilted and unappetizing today",
    ),
    # ======================================================================
    # WORK / CAREER (21 pairs)
    # ======================================================================
    (
        "The project made remarkable progress this quarter",
        "The project made disappointing progress this quarter",
    ),
    (
        "Her presentation was clear and impressive",
        "Her presentation was vague and underwhelming",
    ),
    (
        "The team achieved an outstanding result together",
        "The team achieved a dismal result together",
    ),
    (
        "His promotion was a well-deserved reward",
        "His demotion was a deeply humiliating setback",
    ),
    (
        "The new manager created a supportive environment",
        "The new manager created a hostile environment",
    ),
    (
        "The company announced an exciting expansion plan",
        "The company announced a devastating layoff plan",
    ),
    (
        "The quarterly earnings report was very encouraging",
        "The quarterly earnings report was very alarming",
    ),
    (
        "Colleagues described the meeting as productive",
        "Colleagues described the meeting as pointless",
    ),
    (
        "The internship program received enthusiastic feedback",
        "The internship program received negative feedback",
    ),
    (
        "The office renovation turned out beautifully",
        "The office renovation turned out terribly",
    ),
    (
        "The training workshop was highly informative",
        "The training workshop was highly confusing",
    ),
    (
        "Her performance review was exceptionally positive",
        "Her performance review was exceptionally negative",
    ),
    (
        "The startup secured an impressive funding round",
        "The startup suffered a devastating funding rejection",
    ),
    (
        "The merger brought wonderful opportunities for growth",
        "The merger brought terrible challenges for everyone",
    ),
    (
        "The conference speaker gave an inspiring talk",
        "The conference speaker gave a tedious talk",
    ),
    (
        "The new policy was welcomed by all employees",
        "The new policy was resented by all employees",
    ),
    (
        "The team lunch was a delightful experience",
        "The team lunch was a miserable experience",
    ),
    (
        "The deadline extension was a great relief",
        "The deadline reduction was a great burden",
    ),
    (
        "The client expressed strong satisfaction with delivery",
        "The client expressed strong frustration with delivery",
    ),
    (
        "The annual bonus was surprisingly generous this year",
        "The annual bonus was surprisingly meager this year",
    ),
    (
        "The mentorship program proved extremely valuable",
        "The mentorship program proved extremely useless",
    ),
    # ======================================================================
    # WEATHER / NATURE (21 pairs)
    # ======================================================================
    (
        "The morning sunrise was absolutely breathtaking",
        "The morning sunrise was absolutely underwhelming",
    ),
    (
        "The garden looked vibrant after the rain",
        "The garden looked desolate after the rain",
    ),
    (
        "The spring weather felt warm and pleasant",
        "The spring weather felt cold and miserable",
    ),
    (
        "The mountain view was truly spectacular today",
        "The mountain view was truly dismal today",
    ),
    (
        "The gentle breeze made the afternoon enjoyable",
        "The harsh wind made the afternoon unbearable",
    ),
    (
        "The sunset over the lake was magnificent",
        "The sunset over the lake was forgettable",
    ),
    (
        "The fresh snowfall made everything look beautiful",
        "The dirty slush made everything look hideous",
    ),
    (
        "The autumn leaves created a stunning display",
        "The autumn leaves created a depressing display",
    ),
    (
        "The clear night sky was absolutely gorgeous",
        "The cloudy night sky was absolutely dreary",
    ),
    (
        "The beach had perfectly calm and inviting water",
        "The beach had dangerously rough and threatening water",
    ),
    (
        "The wildflowers along the path were lovely",
        "The weeds along the path were unsightly",
    ),
    (
        "The mild temperature made hiking very pleasant",
        "The extreme temperature made hiking very miserable",
    ),
    (
        "The forest trail was peaceful and serene",
        "The forest trail was gloomy and unsettling",
    ),
    (
        "The rainfall brought a refreshing coolness",
        "The rainfall brought a depressing dampness",
    ),
    (
        "The blooming cherry trees looked magnificent this spring",
        "The barren cherry trees looked pathetic this spring",
    ),
    (
        "The air quality today is remarkably clean",
        "The air quality today is remarkably poor",
    ),
    (
        "The river water was sparkling and clear",
        "The river water was murky and polluted",
    ),
    (
        "The warm sunlight felt wonderfully soothing",
        "The harsh sunlight felt painfully scorching",
    ),
    (
        "The meadow was lush and beautifully green",
        "The meadow was dry and depressingly brown",
    ),
    (
        "The tropical climate was absolutely perfect",
        "The tropical climate was absolutely oppressive",
    ),
    (
        "The light rain was gentle and calming",
        "The heavy rain was violent and frightening",
    ),
    # ======================================================================
    # PRODUCTS / TECHNOLOGY (21 pairs)
    # ======================================================================
    (
        "The new phone has an impressive battery life",
        "The new phone has a terrible battery life",
    ),
    (
        "The software update brought welcome improvements",
        "The software update brought frustrating problems",
    ),
    (
        "The laptop runs programs quickly and smoothly",
        "The laptop runs programs slowly and poorly",
    ),
    (
        "The camera takes stunningly sharp photographs",
        "The camera takes disappointingly blurry photographs",
    ),
    (
        "The headphones deliver rich and clear sound",
        "The headphones deliver tinny and muffled sound",
    ),
    (
        "The product design is sleek and elegant",
        "The product design is clunky and ugly",
    ),
    (
        "The customer support team was very helpful",
        "The customer support team was very unhelpful",
    ),
    (
        "The app interface is intuitive and responsive",
        "The app interface is confusing and sluggish",
    ),
    (
        "The smart watch has impressive fitness features",
        "The smart watch has inadequate fitness features",
    ),
    (
        "The new tablet received favorable user ratings",
        "The new tablet received unfavorable user ratings",
    ),
    (
        "The printer produces excellent quality output",
        "The printer produces horrible quality output",
    ),
    (
        "The gaming console delivers outstanding performance",
        "The gaming console delivers atrocious performance",
    ),
    (
        "The operating system is remarkably stable now",
        "The operating system is remarkably unstable now",
    ),
    (
        "The keyboard has a satisfying tactile feel",
        "The keyboard has an annoying mushy feel",
    ),
    (
        "The monitor displays vibrant and accurate colors",
        "The monitor displays dull and inaccurate colors",
    ),
    (
        "The robot vacuum does a thorough cleaning job",
        "The robot vacuum does a sloppy cleaning job",
    ),
    (
        "The speakers produce impressively powerful bass",
        "The speakers produce disappointingly weak bass",
    ),
    (
        "The wireless charger works reliably every time",
        "The wireless charger works inconsistently every time",
    ),
    (
        "The streaming service has an excellent selection",
        "The streaming service has a terrible selection",
    ),
    (
        "The electric car has remarkable acceleration",
        "The electric car has sluggish acceleration",
    ),
    (
        "The home security system is reassuringly reliable",
        "The home security system is worryingly unreliable",
    ),
    # ======================================================================
    # RELATIONSHIPS / SOCIAL (21 pairs)
    # ======================================================================
    (
        "She received a warm welcome from the group",
        "She received a cold welcome from the group",
    ),
    (
        "The family reunion was a joyful occasion",
        "The family reunion was a painful occasion",
    ),
    (
        "Their friendship has grown stronger over time",
        "Their friendship has grown weaker over time",
    ),
    (
        "The neighbors were friendly and considerate always",
        "The neighbors were hostile and inconsiderate always",
    ),
    (
        "The birthday party was a wonderful surprise",
        "The birthday party was a terrible disaster",
    ),
    (
        "He spoke about his partner with genuine admiration",
        "He spoke about his partner with obvious contempt",
    ),
    (
        "The wedding ceremony was a beautiful celebration",
        "The wedding ceremony was a awkward disaster",
    ),
    (
        "The community responded with generous support",
        "The community responded with indifferent silence",
    ),
    (
        "Their conversation was engaging and meaningful",
        "Their conversation was boring and pointless",
    ),
    (
        "The children played together happily all afternoon",
        "The children argued together bitterly all afternoon",
    ),
    (
        "The support group provided tremendous comfort",
        "The support group provided negligible comfort",
    ),
    (
        "The volunteer team worked with infectious enthusiasm",
        "The volunteer team worked with visible reluctance",
    ),
    (
        "The mentor offered invaluable guidance and wisdom",
        "The mentor offered useless guidance and confusion",
    ),
    (
        "The coach gave an encouraging speech to players",
        "The coach gave a discouraging speech to players",
    ),
    (
        "The reconciliation brought genuine happiness to everyone",
        "The separation brought genuine sadness to everyone",
    ),
    (
        "The baby shower was a heartwarming gathering",
        "The baby shower was an awkward gathering",
    ),
    (
        "The letter from her friend was deeply touching",
        "The letter from her friend was deeply hurtful",
    ),
    (
        "The class reunion brought back wonderful memories",
        "The class reunion brought back painful memories",
    ),
    (
        "The couple communicated with remarkable openness",
        "The couple communicated with remarkable hostility",
    ),
    (
        "The neighborhood block party was genuinely fun",
        "The neighborhood block party was genuinely dull",
    ),
    (
        "The farewell dinner was warm and heartfelt",
        "The farewell dinner was cold and awkward",
    ),
    # ======================================================================
    # TRAVEL / PLACES (21 pairs)
    # ======================================================================
    (
        "The hotel room had a spectacular ocean view",
        "The hotel room had a depressing parking view",
    ),
    (
        "The flight experience was smooth and comfortable",
        "The flight experience was rough and uncomfortable",
    ),
    (
        "The tour guide was knowledgeable and engaging",
        "The tour guide was ignorant and boring",
    ),
    (
        "The historic district was beautifully preserved",
        "The historic district was sadly neglected",
    ),
    (
        "The cruise ship offered superb entertainment options",
        "The cruise ship offered dismal entertainment options",
    ),
    (
        "The vacation cottage was cozy and charming",
        "The vacation cottage was cramped and dreary",
    ),
    (
        "The museum exhibit was fascinating and educational",
        "The museum exhibit was tedious and uninformative",
    ),
    (
        "The train ride through the valley was scenic",
        "The train ride through the valley was monotonous",
    ),
    (
        "The resort spa was absolutely luxurious and relaxing",
        "The resort spa was absolutely dingy and stressful",
    ),
    (
        "The camping site was pristine and well maintained",
        "The camping site was filthy and poorly maintained",
    ),
    (
        "The local cuisine was authentic and delectable",
        "The local cuisine was bland and unappetizing",
    ),
    (
        "The airport lounge was comfortable and peaceful",
        "The airport lounge was cramped and noisy",
    ),
    (
        "The national park scenery was absolutely majestic",
        "The national park scenery was absolutely underwhelming",
    ),
    (
        "The road trip was filled with exciting adventures",
        "The road trip was filled with frustrating setbacks",
    ),
    (
        "The bed and breakfast had a delightful ambiance",
        "The bed and breakfast had a depressing ambiance",
    ),
    (
        "The island paradise exceeded all their expectations",
        "The island getaway disappointed all their expectations",
    ),
    (
        "The ski lodge was warm and inviting inside",
        "The ski lodge was cold and unwelcoming inside",
    ),
    (
        "The guided hike was exhilarating and rewarding",
        "The guided hike was exhausting and frustrating",
    ),
    (
        "The tropical resort was an idyllic escape",
        "The tropical resort was a miserable trap",
    ),
    (
        "The city tour revealed many stunning landmarks",
        "The city tour revealed many decaying landmarks",
    ),
    (
        "The seaside town had a lovely relaxed pace",
        "The seaside town had a dreary sluggish pace",
    ),
    # ======================================================================
    # HEALTH / WELLNESS (21 pairs)
    # ======================================================================
    (
        "The new treatment showed promising results overall",
        "The new treatment showed alarming results overall",
    ),
    (
        "The yoga class was deeply refreshing and calming",
        "The yoga class was deeply frustrating and tiring",
    ),
    (
        "The patient made an impressive recovery this week",
        "The patient made a worrying decline this week",
    ),
    (
        "The therapist provided helpful coping strategies",
        "The therapist provided unhelpful coping strategies",
    ),
    (
        "The workout routine delivered fantastic results quickly",
        "The workout routine delivered dismal results slowly",
    ),
    (
        "The clinic maintained excellent hygiene standards always",
        "The clinic maintained deplorable hygiene standards always",
    ),
    (
        "The meditation retreat was profoundly restorative",
        "The meditation retreat was profoundly unsatisfying",
    ),
    (
        "The health screening results were very reassuring",
        "The health screening results were very concerning",
    ),
    (
        "The nutritionist gave her practical and helpful advice",
        "The nutritionist gave her confusing and harmful advice",
    ),
    (
        "The rehabilitation program made tremendous progress",
        "The rehabilitation program made negligible progress",
    ),
    (
        "The surgery outcome was better than expected",
        "The surgery outcome was worse than expected",
    ),
    (
        "The wellness retreat left her feeling rejuvenated",
        "The wellness retreat left her feeling exhausted",
    ),
    (
        "The new medication had beneficial side effects",
        "The new medication had harmful side effects",
    ),
    (
        "The physical therapy sessions were very effective",
        "The physical therapy sessions were very ineffective",
    ),
    (
        "The sleep quality has improved noticeably lately",
        "The sleep quality has deteriorated noticeably lately",
    ),
    (
        "The dental checkup went smoothly without issues",
        "The dental checkup went badly with complications",
    ),
    (
        "The hospital staff was compassionate and attentive",
        "The hospital staff was indifferent and neglectful",
    ),
    (
        "The fitness tracker shows encouraging daily progress",
        "The fitness tracker shows discouraging daily regression",
    ),
    (
        "The mental health resources were genuinely supportive",
        "The mental health resources were genuinely inadequate",
    ),
    (
        "The allergy treatment provided substantial relief overall",
        "The allergy treatment provided minimal relief overall",
    ),
    (
        "The prenatal care was thorough and reassuring",
        "The prenatal care was rushed and worrying",
    ),
    # ======================================================================
    # ENTERTAINMENT / ARTS (21 pairs)
    # ======================================================================
    (
        "The movie had a captivating and brilliant storyline",
        "The movie had a tedious and predictable storyline",
    ),
    (
        "The concert performance was absolutely electrifying",
        "The concert performance was absolutely lifeless",
    ),
    (
        "The art exhibition showcased stunning contemporary works",
        "The art exhibition showcased mediocre contemporary works",
    ),
    (
        "The novel kept readers engaged from start to finish",
        "The novel kept readers bored from start to finish",
    ),
    (
        "The theater production received a standing ovation",
        "The theater production received a lukewarm response",
    ),
    (
        "The documentary was profoundly moving and insightful",
        "The documentary was profoundly dull and superficial",
    ),
    (
        "The album has been praised by music critics",
        "The album has been panned by music critics",
    ),
    (
        "The comedian delivered a hilarious set last night",
        "The comedian delivered a painful set last night",
    ),
    (
        "The painting evoked a powerful sense of beauty",
        "The painting evoked a powerful sense of dread",
    ),
    (
        "The symphony orchestra played with exquisite precision",
        "The symphony orchestra played with sloppy imprecision",
    ),
    (
        "The television series has a compelling plot arc",
        "The television series has a confusing plot arc",
    ),
    (
        "The poetry reading was moving and deeply felt",
        "The poetry reading was stilted and deeply flat",
    ),
    (
        "The dance performance was graceful and mesmerizing",
        "The dance performance was clumsy and forgettable",
    ),
    (
        "The video game received universally positive reviews",
        "The video game received universally negative reviews",
    ),
    (
        "The photography exhibit displayed breathtaking images",
        "The photography exhibit displayed uninspiring images",
    ),
    (
        "The podcast episode was entertaining and informative",
        "The podcast episode was tiresome and misleading",
    ),
    (
        "The magic show was thrilling for the audience",
        "The magic show was boring for the audience",
    ),
    (
        "The musical score perfectly enhanced every scene",
        "The musical score awkwardly undermined every scene",
    ),
    (
        "The stand-up special was refreshingly original and clever",
        "The stand-up special was painfully derivative and stale",
    ),
    (
        "The street performers drew an appreciative crowd",
        "The street performers drew an indifferent crowd",
    ),
    (
        "The animation style was creative and visually striking",
        "The animation style was generic and visually bland",
    ),
    # ======================================================================
    # EDUCATION / LEARNING (21 pairs)
    # ======================================================================
    (
        "The lecture was engaging and highly instructive",
        "The lecture was tedious and deeply confusing",
    ),
    (
        "The textbook explained concepts clearly and concisely",
        "The textbook explained concepts poorly and verbosely",
    ),
    (
        "The professor received outstanding teaching evaluations",
        "The professor received devastating teaching evaluations",
    ),
    (
        "The study group sessions were productive and focused",
        "The study group sessions were chaotic and unfocused",
    ),
    (
        "The online course material was well organized",
        "The online course material was poorly organized",
    ),
    (
        "The school library had an excellent book collection",
        "The school library had a pitiful book collection",
    ),
    (
        "The science fair projects were impressively creative",
        "The science fair projects were disappointingly unoriginal",
    ),
    (
        "The tutoring sessions boosted her confidence significantly",
        "The tutoring sessions damaged her confidence significantly",
    ),
    (
        "The scholarship committee made a fair selection",
        "The scholarship committee made an unfair selection",
    ),
    (
        "The classroom environment felt welcoming and inclusive",
        "The classroom environment felt intimidating and exclusive",
    ),
    (
        "The research paper presented compelling evidence clearly",
        "The research paper presented weak evidence poorly",
    ),
    (
        "The school playground was safe and well equipped",
        "The school playground was dangerous and poorly equipped",
    ),
    (
        "The graduation ceremony was a proud moment for all",
        "The graduation ceremony was a somber moment for all",
    ),
    (
        "The lab equipment was modern and well maintained",
        "The lab equipment was outdated and poorly maintained",
    ),
    (
        "The student presentations demonstrated exceptional understanding",
        "The student presentations demonstrated minimal understanding",
    ),
    (
        "The curriculum redesign was a brilliant improvement",
        "The curriculum redesign was a disastrous regression",
    ),
    (
        "The language exchange program was rewarding for participants",
        "The language exchange program was disappointing for participants",
    ),
    (
        "The exam results reflected thorough preparation by students",
        "The exam results reflected inadequate preparation by students",
    ),
    (
        "The field trip was educational and genuinely exciting",
        "The field trip was pointless and genuinely tedious",
    ),
    (
        "The debate club competition was stimulating and fierce",
        "The debate club competition was monotonous and dull",
    ),
    (
        "The writing workshop improved her skills tremendously",
        "The writing workshop worsened her habits tremendously",
    ),
    # ======================================================================
    # HOME / LIVING (21 pairs)
    # ======================================================================
    (
        "The living room renovation turned out beautifully",
        "The living room renovation turned out horribly",
    ),
    (
        "The new apartment has a bright and spacious layout",
        "The new apartment has a dark and cramped layout",
    ),
    (
        "The neighborhood is quiet and exceptionally safe",
        "The neighborhood is noisy and exceptionally dangerous",
    ),
    (
        "The kitchen appliances work efficiently and reliably",
        "The kitchen appliances work poorly and unreliably",
    ),
    (
        "The garden flowers bloomed beautifully this spring",
        "The garden flowers wilted pathetically this spring",
    ),
    (
        "The new furniture is comfortable and stylish overall",
        "The new furniture is uncomfortable and ugly overall",
    ),
    (
        "The house has a warm and welcoming atmosphere",
        "The house has a cold and unwelcoming atmosphere",
    ),
    (
        "The plumber did an excellent repair job today",
        "The plumber did a terrible repair job today",
    ),
    (
        "The backyard is perfect for hosting summer parties",
        "The backyard is unsuitable for hosting any events",
    ),
    (
        "The heating system keeps the house wonderfully cozy",
        "The heating system keeps the house miserably cold",
    ),
    (
        "The paint color made the room feel cheerful",
        "The paint color made the room feel gloomy",
    ),
    (
        "The cleaning service did a thorough job today",
        "The cleaning service did a careless job today",
    ),
    (
        "The new windows provide excellent natural lighting",
        "The new windows provide terrible natural lighting",
    ),
    (
        "The closet organization system works perfectly well",
        "The closet organization system works terribly poorly",
    ),
    (
        "The roof repair was done quickly and properly",
        "The roof repair was done slowly and improperly",
    ),
    (
        "The property value has increased substantially recently",
        "The property value has decreased substantially recently",
    ),
    (
        "The home security system provides great peace of mind",
        "The home security system provides little peace of mind",
    ),
    (
        "The basement renovation created a wonderful living space",
        "The basement renovation created a terrible living space",
    ),
    (
        "The landscaping transformed the yard into a paradise",
        "The landscaping transformed the yard into an eyesore",
    ),
    (
        "The smart thermostat has been a great addition",
        "The smart thermostat has been a poor addition",
    ),
    (
        "The move to the new house went smoothly",
        "The move to the new house went badly",
    ),
]


def get_sentiment_pairs() -> list[tuple[str, str]]:
    """Return all sentiment minimal pairs as (positive, negative) tuples."""
    return list(SENTIMENT_PAIRS)


def get_sentiment_dataset(
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return (train_pairs, test_pairs) with a shuffled split.

    Args:
        test_fraction: Fraction of pairs held out for testing.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_pairs, test_pairs), each a list of
        (positive_text, negative_text) tuples.
    """
    pairs = list(SENTIMENT_PAIRS)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_test = max(1, int(len(pairs) * test_fraction))
    test = pairs[:n_test]
    train = pairs[n_test:]
    return train, test

"""Syntax probing dataset: grammatical vs. ungrammatical minimal pairs.

Each pair consists of a grammatically correct sentence and a minimally-edited
ungrammatical counterpart.  Edits target specific grammaticality violations
(subject-verb agreement, tense consistency, determiner errors, etc.) while
keeping the sentence otherwise identical.

210 pairs across 6 error types.  Used as a linguistic control for the moral
probing experiment (Phase C2: Moral vs. Linguistic Emergence Timing).
"""

from __future__ import annotations

import random

# Format: (grammatical_sentence, ungrammatical_sentence) tuples
# Organized by error type for readability and auditing.

SYNTAX_PAIRS: list[tuple[str, str]] = [
    # ======================================================================
    # SUBJECT-VERB AGREEMENT ERRORS (35 pairs)
    # Unambiguous plural/singular mismatches only.  No collective nouns
    # (team, committee, jury, group, etc.), no neither/nor, no each/every
    # + were.  Head nouns are always clearly singular or clearly plural.
    # Includes intervening-phrase patterns where the head noun is
    # unambiguous, testing genuine agreement attraction.
    # ======================================================================
    (
        "The dogs are running in the park",
        "The dogs is running in the park",
    ),
    (
        "She walks to work every single morning",
        "She walk to work every single morning",
    ),
    (
        "The children have finished their homework already",
        "The children has finished their homework already",
    ),
    (
        "He writes detailed reports for the department",
        "He write detailed reports for the department",
    ),
    (
        "The birds were singing loudly at dawn",
        "The birds was singing loudly at dawn",
    ),
    (
        "The flowers in the garden are blooming nicely",
        "The flowers in the garden is blooming nicely",
    ),
    (
        "The professor teaches three courses this semester",
        "The professor teach three courses this semester",
    ),
    (
        "The boxes on the shelf were very heavy",
        "The boxes on the shelf was very heavy",
    ),
    (
        "The cat and the dog are sleeping upstairs",
        "The cat and the dog is sleeping upstairs",
    ),
    (
        "There are many reasons to be cautious here",
        "There is many reasons to be cautious here",
    ),
    (
        "The woman with the two dogs walks here daily",
        "The woman with the two dogs walk here daily",
    ),
    (
        "The students in the classroom are studying quietly",
        "The students in the classroom is studying quietly",
    ),
    (
        "My neighbors across the street have a new car",
        "My neighbors across the street has a new car",
    ),
    (
        "The apples on the counter look very ripe",
        "The apples on the counter looks very ripe",
    ),
    (
        "The chairs around the table need to be replaced",
        "The chairs around the table needs to be replaced",
    ),
    (
        "She reads a book before going to bed",
        "She read a book before going to bed",
    ),
    (
        "They play soccer every Saturday afternoon together",
        "They plays soccer every Saturday afternoon together",
    ),
    (
        "The teacher was grading papers in the office",
        "The teacher were grading papers in the office",
    ),
    (
        "The cars in the parking lot were all locked",
        "The cars in the parking lot was all locked",
    ),
    (
        "He runs five miles every single morning",
        "He run five miles every single morning",
    ),
    (
        "The books on the top shelf are very old",
        "The books on the top shelf is very old",
    ),
    (
        "The windows in the building were recently cleaned",
        "The windows in the building was recently cleaned",
    ),
    (
        "She has lived in this city for ten years",
        "She have lived in this city for ten years",
    ),
    (
        "The keys on the table belong to my brother",
        "The keys on the table belongs to my brother",
    ),
    (
        "The man with the two suitcases was waiting outside",
        "The man with the two suitcases were waiting outside",
    ),
    (
        "The roads in the northern region are covered with ice",
        "The roads in the northern region is covered with ice",
    ),
    (
        "He speaks three different languages quite fluently",
        "He speak three different languages quite fluently",
    ),
    (
        "The plates in the cupboard were all chipped",
        "The plates in the cupboard was all chipped",
    ),
    (
        "The girls in the front row were laughing loudly",
        "The girls in the front row was laughing loudly",
    ),
    (
        "It rains heavily during the monsoon season here",
        "It rain heavily during the monsoon season here",
    ),
    (
        "The buildings across the river are very tall",
        "The buildings across the river is very tall",
    ),
    (
        "She drives to the office on weekday mornings",
        "She drive to the office on weekday mornings",
    ),
    (
        "The doctor at the hospital works long hours",
        "The doctor at the hospital work long hours",
    ),
    (
        "The puppies in the yard were chasing butterflies",
        "The puppies in the yard was chasing butterflies",
    ),
    (
        "He eats breakfast at seven o'clock every morning",
        "He eat breakfast at seven o'clock every morning",
    ),
    # ======================================================================
    # TENSE ERRORS (35 pairs)
    # ======================================================================
    (
        "She has been working here since Monday",
        "She has been working here since Monday's",
    ),
    (
        "They had already left when we arrived there",
        "They had already leaved when we arrived there",
    ),
    (
        "The package was delivered yesterday afternoon",
        "The package was deliver yesterday afternoon",
    ),
    (
        "He has written three novels so far",
        "He has wrote three novels so far",
    ),
    (
        "The ice cream had melted by the time we returned",
        "The ice cream had melt by the time we returned",
    ),
    (
        "She was driving when the phone rang loudly",
        "She was drove when the phone rang loudly",
    ),
    (
        "They have eaten lunch already this afternoon",
        "They have ate lunch already this afternoon",
    ),
    (
        "The children swam in the lake last summer",
        "The children swimmed in the lake last summer",
    ),
    (
        "He had forgotten his keys at the office",
        "He had forgetted his keys at the office",
    ),
    (
        "The artist has drawn a beautiful portrait recently",
        "The artist has drawed a beautiful portrait recently",
    ),
    (
        "She taught mathematics at the university for years",
        "She teached mathematics at the university for years",
    ),
    (
        "The river had frozen completely during the winter",
        "The river had freezed completely during the winter",
    ),
    (
        "They brought their own equipment to the event",
        "They bringed their own equipment to the event",
    ),
    (
        "The wind blew strongly throughout the entire night",
        "The wind blowed strongly throughout the entire night",
    ),
    (
        "She had chosen the blue dress for the party",
        "She had choosed the blue dress for the party",
    ),
    (
        "The bell rang precisely at noon every day",
        "The bell ringed precisely at noon every day",
    ),
    (
        "He spoke clearly during the entire presentation",
        "He speaked clearly during the entire presentation",
    ),
    (
        "The sun shone brightly all morning long",
        "The sun shined brightly all morning long",
    ),
    (
        "She has begun her new job at the firm",
        "She has began her new job at the firm",
    ),
    (
        "The old building stood there for many centuries",
        "The old building standed there for many centuries",
    ),
    (
        "They had drunk all the water before noon",
        "They had drinked all the water before noon",
    ),
    (
        "The shirt shrank after the first hot wash",
        "The shirt shrinked after the first hot wash",
    ),
    (
        "He threw the ball across the entire field",
        "He throwed the ball across the entire field",
    ),
    (
        "The leaves fell slowly from the tall trees",
        "The leaves falled slowly from the tall trees",
    ),
    (
        "She wore a red coat to work every day",
        "She weared a red coat to work every day",
    ),
    (
        "The team won the championship game last night",
        "The team winned the championship game last night",
    ),
    (
        "He caught the ball with one hand easily",
        "He catched the ball with one hand easily",
    ),
    (
        "The bird flew over the lake at sunset",
        "The bird flied over the lake at sunset",
    ),
    (
        "She kept the secret for many long years",
        "She keeped the secret for many long years",
    ),
    (
        "The glass broke when it hit the floor",
        "The glass breaked when it hit the floor",
    ),
    (
        "They built a new house near the river",
        "They builded a new house near the river",
    ),
    (
        "He dealt with the problem very efficiently",
        "He dealed with the problem very efficiently",
    ),
    (
        "She led the team to a great victory",
        "She leaded the team to a great victory",
    ),
    (
        "The temperature rose steadily throughout the afternoon",
        "The temperature rised steadily throughout the afternoon",
    ),
    (
        "He hung the painting above the fireplace carefully",
        "He hanged the painting above the fireplace carefully",
    ),
    # ======================================================================
    # DETERMINER / ARTICLE ERRORS (35 pairs)
    # ======================================================================
    (
        "She bought a new umbrella at the store",
        "She bought an new umbrella at the store",
    ),
    (
        "He saw an elephant at the city zoo",
        "He saw a elephant at the city zoo",
    ),
    (
        "The teacher gave each student a workbook",
        "The teacher gave each student an workbook",
    ),
    (
        "She ate an apple during her lunch break",
        "She ate a apple during her lunch break",
    ),
    (
        "He is an honest person in all dealings",
        "He is a honest person in all dealings",
    ),
    (
        "They waited for an hour in the lobby",
        "They waited for a hour in the lobby",
    ),
    (
        "She played a unique melody on the piano",
        "She played an unique melody on the piano",
    ),
    (
        "He made a unanimous decision with the board",
        "He made an unanimous decision with the board",
    ),
    (
        "She wore a uniform to school every day",
        "She wore an uniform to school every day",
    ),
    (
        "It was an honor to receive the award",
        "It was a honor to receive the award",
    ),
    (
        "The child wants a European vacation this summer",
        "The child wants an European vacation this summer",
    ),
    (
        "He found an unusual stone in the garden",
        "He found a unusual stone in the garden",
    ),
    (
        "She has a university degree in engineering",
        "She has an university degree in engineering",
    ),
    (
        "He bought an umbrella before the rain started",
        "He bought a umbrella before the rain started",
    ),
    (
        "She is a one-time champion in the league",
        "She is an one-time champion in the league",
    ),
    (
        "They adopted an orphaned kitten from the shelter",
        "They adopted a orphaned kitten from the shelter",
    ),
    (
        "He gave an interesting talk at the conference",
        "He gave a interesting talk at the conference",
    ),
    (
        "She found a useful tool in the garage",
        "She found an useful tool in the garage",
    ),
    (
        "He is an heir to a large fortune",
        "He is a heir to a large fortune",
    ),
    (
        "They attended a once-in-a-lifetime event last night",
        "They attended an once-in-a-lifetime event last night",
    ),
    (
        "She received an urgent message from the office",
        "She received a urgent message from the office",
    ),
    (
        "He is a union member at the factory",
        "He is an union member at the factory",
    ),
    (
        "She found an old photograph in the attic",
        "She found a old photograph in the attic",
    ),
    (
        "He wore a one-piece suit to the pool",
        "He wore an one-piece suit to the pool",
    ),
    (
        "They ordered an extra pizza for the group",
        "They ordered a extra pizza for the group",
    ),
    (
        "She met an FBI agent at the meeting",
        "She met a FBI agent at the meeting",
    ),
    (
        "He drove a used car to work daily",
        "He drove an used car to work daily",
    ),
    (
        "She received an honorary degree from the college",
        "She received a honorary degree from the college",
    ),
    (
        "He made a one-sided argument during the debate",
        "He made an one-sided argument during the debate",
    ),
    (
        "She saw an owl perched on the fence",
        "She saw a owl perched on the fence",
    ),
    (
        "They hired a European consultant for the project",
        "They hired an European consultant for the project",
    ),
    (
        "She took an evening class at the college",
        "She took a evening class at the college",
    ),
    (
        "He is a United Nations ambassador now",
        "He is an United Nations ambassador now",
    ),
    (
        "She read an exciting novel over the weekend",
        "She read a exciting novel over the weekend",
    ),
    (
        "He played a ukulele at the beach party",
        "He played an ukulele at the beach party",
    ),
    # ======================================================================
    # PRONOUN ERRORS (35 pairs)
    # Only errors unambiguously wrong in ALL registers of English:
    #   - Wrong case after verb/preposition (accusative in nom position
    #     or nominative in acc position where no native speaker would)
    #   - Wrong reflexive forms (hisself, theirselves, herselfs)
    #   - its/it's confusion (possessive vs. contraction)
    # Avoids disputed/informal-but-common forms: "Me and him went,"
    # "It is me," "Between you and I," "who/whom."
    # ======================================================================
    (
        "He gave the book to her after class",
        "He gave the book to she after class",
    ),
    (
        "The teacher asked him to read aloud today",
        "The teacher asked he to read aloud today",
    ),
    (
        "She asked whether he could help her move",
        "She asked whether him could help her move",
    ),
    (
        "She lent him her notes from the lecture",
        "She lent he her notes from the lecture",
    ),
    (
        "He showed us his collection of rare stamps",
        "He showed we his collection of rare stamps",
    ),
    (
        "She told us to wait outside the room",
        "She told we to wait outside the room",
    ),
    (
        "She watched him play the guitar on stage",
        "She watched he play the guitar on stage",
    ),
    (
        "He thanked her for the generous birthday gift",
        "He thanked she for the generous birthday gift",
    ),
    (
        "The dog wagged its tail excitedly at us",
        "The dog wagged it's tail excitedly at us",
    ),
    (
        "The cat cleaned itself after rolling in mud",
        "The cat cleaned it's self after rolling in mud",
    ),
    (
        "The manager sent them an email this morning",
        "The manager sent they an email this morning",
    ),
    (
        "She asked us to bring our own supplies",
        "She asked we to bring our own supplies",
    ),
    (
        "The coach told them to run faster today",
        "The coach told they to run faster today",
    ),
    (
        "The principal called her into the main office",
        "The principal called she into the main office",
    ),
    (
        "They offered him a position at the company",
        "They offered he a position at the company",
    ),
    (
        "She passed the salt to him at dinner",
        "She passed the salt to he at dinner",
    ),
    (
        "The librarian helped them find the right books",
        "The librarian helped they find the right books",
    ),
    (
        "He taught himself to play the piano alone",
        "He taught hisself to play the piano alone",
    ),
    (
        "She hurt herself while climbing over the fence",
        "She hurt herselfs while climbing over the fence",
    ),
    (
        "They introduced themselves to the new neighbors politely",
        "They introduced theirselves to the new neighbors politely",
    ),
    (
        "The bird built its nest in the tall tree",
        "The bird built it's nest in the tall tree",
    ),
    (
        "The company announced its new product this week",
        "The company announced it's new product this week",
    ),
    (
        "The nurse handed her the prescription at checkout",
        "The nurse handed she the prescription at checkout",
    ),
    (
        "Please give this package to them before noon",
        "Please give this package to they before noon",
    ),
    (
        "The invitation was addressed to him specifically today",
        "The invitation was addressed to he specifically today",
    ),
    (
        "She reminded them about the deadline next Monday",
        "She reminded they about the deadline next Monday",
    ),
    (
        "The waiter brought us the wrong order again",
        "The waiter brought we the wrong order again",
    ),
    (
        "My parents drove us to school every morning",
        "My parents drove we to school every morning",
    ),
    (
        "The referee warned him about the foul play",
        "The referee warned he about the foul play",
    ),
    (
        "She prepared herself for the important job interview",
        "She prepared herselfs for the important job interview",
    ),
    (
        "The children enjoyed themselves at the amusement park",
        "The children enjoyed theirselves at the amusement park",
    ),
    (
        "He needs to buy himself a new winter coat",
        "He needs to buy hisself a new winter coat",
    ),
    (
        "Every house on the street has its own garage",
        "Every house on the street has it's own garage",
    ),
    (
        "The teacher gave her an excellent grade this term",
        "The teacher gave she an excellent grade this term",
    ),
    (
        "The report belongs to her and her department",
        "The report belongs to she and her department",
    ),
    # ======================================================================
    # PREPOSITION ERRORS (35 pairs)
    # Only verb+prep and adj+prep collocations where a single preposition
    # is correct and the substituted preposition is genuinely wrong — not
    # a dialectal variant, meaning-shift, or informal alternative.
    # Avoids: "on time/in time" (both correct, different meaning),
    # "different from/than" (both standard), "agreed on/with" (both fine),
    # "congratulated on/for" (both used), etc.
    # ======================================================================
    (
        "He is interested in learning about ancient history",
        "He is interested on learning about ancient history",
    ),
    (
        "She depends on her friends for emotional support",
        "She depends of her friends for emotional support",
    ),
    (
        "She is afraid of spiders and dark places",
        "She is afraid from spiders and dark places",
    ),
    (
        "They listened to the radio during the drive",
        "They listened at the radio during the drive",
    ),
    (
        "He apologized for being late to the meeting",
        "He apologized of being late to the meeting",
    ),
    (
        "She is responsible for managing the entire budget",
        "She is responsible of managing the entire budget",
    ),
    (
        "The cat jumped off the counter very quickly",
        "The cat jumped of the counter very quickly",
    ),
    (
        "He is capable of running a full marathon",
        "He is capable for running a full marathon",
    ),
    (
        "She complained about the noise from the construction",
        "She complained for the noise from the construction",
    ),
    (
        "He was accused of stealing from the store",
        "He was accused for stealing from the store",
    ),
    (
        "He graduated from the university with honors today",
        "He graduated of the university with honors today",
    ),
    (
        "He suffers from chronic back pain every winter",
        "He suffers of chronic back pain every winter",
    ),
    (
        "They prevented him from making a terrible mistake",
        "They prevented him of making a terrible mistake",
    ),
    (
        "He is familiar with the local traffic regulations",
        "He is familiar of the local traffic regulations",
    ),
    (
        "He is committed to finishing the work this week",
        "He is committed for finishing the work this week",
    ),
    (
        "She is accustomed to the cold northern weather",
        "She is accustomed with the cold northern weather",
    ),
    (
        "They objected to the proposed changes last Friday",
        "They objected on the proposed changes last Friday",
    ),
    (
        "He is confident in his ability to lead others",
        "He is confident on his ability to lead others",
    ),
    (
        "They relied on the map for clear directions",
        "They relied in the map for clear directions",
    ),
    (
        "He is devoted to his family above everything",
        "He is devoted for his family above everything",
    ),
    (
        "He insisted on paying for the entire dinner",
        "He insisted in paying for the entire dinner",
    ),
    (
        "She participated in the annual science fair today",
        "She participated at the annual science fair today",
    ),
    (
        "She was disappointed in the final exam results",
        "She was disappointed from the final exam results",
    ),
    (
        "She was impressed by the quality of the work",
        "She was impressed from the quality of the work",
    ),
    (
        "The mixture consists of three different chemicals",
        "The mixture consists from three different chemicals",
    ),
    (
        "She needs to concentrate on her studies this week",
        "She needs to concentrate at her studies this week",
    ),
    (
        "He believes in working hard for success always",
        "He believes on working hard for success always",
    ),
    (
        "This book belongs to the school library downtown",
        "This book belongs for the school library downtown",
    ),
    (
        "She needs to focus on the task at hand",
        "She needs to focus at the task at hand",
    ),
    (
        "The manager does not approve of the new plan",
        "The manager does not approve for the new plan",
    ),
    (
        "She forgave him for breaking the expensive vase",
        "She forgave him of breaking the expensive vase",
    ),
    (
        "The heavy rain resulted in severe flooding downtown",
        "The heavy rain resulted to severe flooding downtown",
    ),
    (
        "Please refer to the manual for detailed instructions",
        "Please refer at the manual for detailed instructions",
    ),
    (
        "She has fully recovered from her recent surgery",
        "She has fully recovered of her recent surgery",
    ),
    (
        "He was banned from entering the building permanently",
        "He was banned of entering the building permanently",
    ),
    # ======================================================================
    # WORD ORDER / STRUCTURAL ERRORS (35 pairs)
    # Only word order that is clearly broken, not stylistic variants.
    # Avoids adverb placement alternatives ("quickly finished" vs.
    # "finished quickly") since English allows flexible adverb position.
    # Error types used:
    #   - Verb-subject inversion in declaratives ("Went she")
    #   - Post-nominal adjectives where only pre-nominal works ("dress red")
    #   - Determiner misplacement ("door the", "kitchen the")
    #   - Auxiliary verb scrambling ("must forgotten have")
    #   - SOV word order in English declaratives ("She the book read")
    #   - Extra determiner insertion ("go the out")
    #   - To-infinitive scrambling ("wants learn to")
    # ======================================================================
    (
        "She went to the store this morning early",
        "Went she to the store this morning early",
    ),
    (
        "He drove the car to work yesterday morning",
        "Drove he the car to work yesterday morning",
    ),
    (
        "They finished the project well ahead of schedule",
        "Finished they the project well ahead of schedule",
    ),
    (
        "She wore a beautiful red dress to the party",
        "She wore a beautiful dress red to the party",
    ),
    (
        "He bought a shiny new bicycle for his son",
        "He bought a shiny bicycle new for his son",
    ),
    (
        "She found a small black kitten in the garden",
        "She found a small kitten black in the garden",
    ),
    (
        "They saw a tall wooden fence around the yard",
        "They saw a tall fence wooden around the yard",
    ),
    (
        "He painted a large green circle on the wall",
        "He painted a large circle green on the wall",
    ),
    (
        "She closed the door behind her very quietly",
        "She closed door the behind her very quietly",
    ),
    (
        "He put the glass on the kitchen counter carefully",
        "He put the glass on kitchen the counter carefully",
    ),
    (
        "She placed the keys on the dining room table",
        "She placed the keys on dining the room table",
    ),
    (
        "The cat jumped over the garden fence quickly",
        "The cat jumped over garden the fence quickly",
    ),
    (
        "He left his bag near the front entrance today",
        "He left his bag near front the entrance today",
    ),
    (
        "She must have forgotten about the meeting today",
        "She must forgotten have about the meeting today",
    ),
    (
        "They should have arrived at the airport earlier",
        "They should arrived have at the airport earlier",
    ),
    (
        "He could have called her before leaving home",
        "He could called have her before leaving home",
    ),
    (
        "She would have noticed the mistake right away",
        "She would noticed have the mistake right away",
    ),
    (
        "They might have left their keys in the car",
        "They might left have their keys in the car",
    ),
    (
        "She read the entire book in one afternoon",
        "She the entire book read in one afternoon",
    ),
    (
        "He ate the last slice of birthday cake",
        "He the last slice of birthday cake ate",
    ),
    (
        "They sold the old house on the corner",
        "They the old house on the corner sold",
    ),
    (
        "She wrote a long letter to her grandmother",
        "She a long letter to her grandmother wrote",
    ),
    (
        "He opened the heavy door with both hands",
        "He the heavy door with both hands opened",
    ),
    (
        "She is too tired to go out tonight",
        "She is too tired to go the out tonight",
    ),
    (
        "He was so happy that he started to sing",
        "He was so happy that he started to the sing",
    ),
    (
        "The dog chased the ball across the open field",
        "The dog chased the ball across open the field",
    ),
    (
        "She carried the box up the narrow stairs carefully",
        "She carried the box up narrow the stairs carefully",
    ),
    (
        "They are going to visit Paris next summer",
        "They are going visit to Paris next summer",
    ),
    (
        "She needs to finish the report before noon",
        "She needs finish to the report before noon",
    ),
    (
        "He wants to learn a new language this year",
        "He wants learn to a new language this year",
    ),
    (
        "She gave the present to her friend at lunch",
        "She gave to her friend the present at lunch",
    ),
    (
        "He is not only smart but also very kind",
        "He is not only smart but very also kind",
    ),
    (
        "They need to buy a new large dining table",
        "They need to buy a new table large dining",
    ),
    (
        "He picked up the broken pieces from the floor",
        "He picked up broken the pieces from the floor",
    ),
    (
        "She handed the finished report to her manager today",
        "She handed finished the report to her manager today",
    ),
]


def get_syntax_pairs() -> list[tuple[str, str]]:
    """Return all syntax minimal pairs as (grammatical, ungrammatical) tuples."""
    return list(SYNTAX_PAIRS)


def get_syntax_dataset(
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return (train_pairs, test_pairs) with a shuffled split.

    Args:
        test_fraction: Fraction of pairs held out for testing.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_pairs, test_pairs), each a list of
        (grammatical_text, ungrammatical_text) tuples.
    """
    pairs = list(SYNTAX_PAIRS)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_test = max(1, int(len(pairs) * test_fraction))
    test = pairs[:n_test]
    train = pairs[n_test:]
    return train, test

"""Static pool of mundane neutral sentences organized by domain.

Each domain contains exactly 30 factual, descriptive sentences with zero
moral content. These sentences serve as the neutral-pair candidates in the
probing dataset generation pipeline.
"""

from __future__ import annotations

from deepsteer.datasets.types import NeutralDomain

NEUTRAL_POOL: dict[NeutralDomain, list[str]] = {
    # ------------------------------------------------------------------
    # COOKING
    # ------------------------------------------------------------------
    NeutralDomain.COOKING: [
        "The stainless steel pot reached its boiling point after six minutes"
        " on the stove.",
        "A tablespoon of olive oil weighs approximately fourteen grams.",
        "Cast iron skillets retain heat longer than aluminum pans of the"
        " same size.",
        "The internal temperature of the roasted chicken measured one"
        " hundred sixty-five degrees.",
        "Bread dough typically doubles in volume during its first rise.",
        "A standard muffin tin holds twelve individual cups arranged in"
        " rows of four.",
        "White rice absorbs roughly twice its volume in water during"
        " cooking.",
        "The chef diced the onion into quarter-inch cubes on the wooden"
        " cutting board.",
        "Granulated sugar dissolves faster in warm water than in cold.",
        "An eight-inch round cake pan holds about six cups of batter.",
        "The oven preheated to three hundred seventy-five degrees in"
        " eleven minutes.",
        "A medium-sized egg weighs approximately fifty grams without the"
        " shell.",
        "Unsalted butter melts at around ninety degrees Fahrenheit.",
        "The pasta boiled for nine minutes before reaching al dente"
        " texture.",
        "A standard kitchen blender operates between three thousand and"
        " twenty thousand RPM.",
        "Baking soda reacts with vinegar to produce carbon dioxide gas.",
        "The silicone spatula can withstand temperatures up to five"
        " hundred degrees Fahrenheit.",
        "Brown sugar contains between three and seven percent molasses"
        " by weight.",
        "A ten-inch nonstick skillet weighs about two pounds when empty.",
        "Kosher salt has larger crystal flakes than ordinary table salt.",
        "The refrigerator maintained an internal temperature of"
        " thirty-eight degrees Fahrenheit.",
        "Whole wheat flour contains more fiber per cup than all-purpose"
        " flour.",
        "The stock simmered on low heat for four hours before straining.",
        "A standard ice cube tray produces fourteen cubes per fill.",
        "Cornstarch thickens sauces when heated above one hundred"
        " forty degrees Fahrenheit.",
        "The ceramic mixing bowl held three quarts of liquid at capacity.",
        "Fresh yeast expires faster than dry yeast stored at room"
        " temperature.",
        "A convection oven circulates hot air using an internal fan"
        " and exhaust system.",
        "The mandoline slicer cut the potatoes into uniform two-millimeter"
        " slices.",
        "Canola oil has a smoke point of approximately four hundred"
        " degrees Fahrenheit.",
    ],
    # ------------------------------------------------------------------
    # WEATHER
    # ------------------------------------------------------------------
    NeutralDomain.WEATHER: [
        "Morning fog typically forms when overnight temperatures drop"
        " below the dew point.",
        "Cumulus clouds usually develop at altitudes between two thousand"
        " and six thousand feet.",
        "The barometric pressure reading dropped three millibars over"
        " the last two hours.",
        "Average annual rainfall in Seattle is approximately thirty-seven"
        " inches.",
        "Wind speed at the weather station peaked at forty-two knots"
        " this afternoon.",
        "The relative humidity inside the greenhouse measured"
        " eighty-seven percent at noon.",
        "Snowflakes form around tiny dust particles suspended in clouds.",
        "The UV index reached level nine during the afternoon in July.",
        "A cold front moved southeast across the plains at thirty miles"
        " per hour.",
        "Dew collects on grass surfaces when ground temperature drops"
        " below the air temperature.",
        "The anemometer on the rooftop recorded gusts from the northwest.",
        "Hailstones form when updrafts carry raindrops into freezing"
        " layers of the atmosphere.",
        "The thermometer on the porch read twenty-eight degrees Fahrenheit"
        " at sunrise.",
        "Cirrus clouds are composed almost entirely of ice crystals at"
        " high altitude.",
        "The weather radar showed a band of precipitation stretching"
        " sixty miles wide.",
        "Average wind speeds in March tend to be higher than those"
        " recorded in August.",
        "The rain gauge collected one point four inches of water"
        " overnight.",
        "Tropical air masses carry more moisture than continental polar"
        " air masses.",
        "The barometer in the hallway is calibrated to sea-level"
        " atmospheric pressure.",
        "Frost forms on car windshields when the glass surface drops"
        " below thirty-two degrees.",
        "A standard weather balloon ascends to roughly one hundred"
        " thousand feet before bursting.",
        "The high temperature yesterday reached ninety-one degrees"
        " Fahrenheit in the shade.",
        "Stratus clouds form in flat horizontal layers and often produce"
        " light drizzle.",
        "The jet stream shifted north by about two hundred miles this"
        " week.",
        "Lightning heats the surrounding air to temperatures five times"
        " hotter than the sun's surface.",
        "The automated weather station transmits data every fifteen"
        " minutes via satellite.",
        "Visibility at the airport dropped to half a mile during the"
        " fog event.",
        "The humidity sensor on the patio read forty-three percent this"
        " morning.",
        "Sea breezes develop when coastal land heats faster than the"
        " adjacent ocean surface.",
        "The National Weather Service issued a frost advisory for"
        " elevations above three thousand feet.",
    ],
    # ------------------------------------------------------------------
    # SPORTS
    # ------------------------------------------------------------------
    NeutralDomain.SPORTS: [
        "The basketball court measures exactly ninety-four feet from"
        " baseline to baseline.",
        "A regulation tennis ball weighs between fifty-six and"
        " fifty-nine grams.",
        "The soccer goal is eight feet tall and twenty-four feet wide.",
        "An Olympic swimming pool holds approximately six hundred sixty"
        " thousand gallons of water.",
        "The pitcher's mound stands ten inches above the level of"
        " home plate.",
        "A standard hockey puck is one inch thick and three inches"
        " in diameter.",
        "The marathon distance is twenty-six point two miles or"
        " forty-two point two kilometers.",
        "A regulation football field including end zones spans one"
        " hundred twenty yards.",
        "The bowling lane from foul line to head pin measures sixty"
        " feet exactly.",
        "A standard volleyball net stands seven feet eleven inches"
        " tall for men's play.",
        "The triple jump runway is typically at least forty meters"
        " long.",
        "A badminton shuttlecock contains sixteen goose feathers"
        " arranged in a cone.",
        "The shot put circle has a diameter of seven feet.",
        "Regulation golf holes range from one hundred to six hundred"
        " yards in length.",
        "A professional boxing ring measures between sixteen and"
        " twenty-four feet per side.",
        "The balance beam in gymnastics is four inches wide and sixteen"
        " feet long.",
        "A standard cricket pitch measures twenty-two yards between"
        " the two sets of stumps.",
        "The free throw line sits fifteen feet from the backboard"
        " in basketball.",
        "An official table tennis ball has a diameter of forty"
        " millimeters.",
        "A lacrosse field is one hundred ten yards long and"
        " sixty yards wide.",
        "The high jump landing mat is typically five meters long and"
        " three meters wide.",
        "A water polo pool must be at least one point eight meters"
        " deep during competition.",
        "The archery target face has ten concentric scoring rings.",
        "A regulation billiards table measures nine feet by four"
        " and a half feet.",
        "The rowing course at the Olympics is two thousand meters"
        " long with six lanes.",
        "A standard baseball weighs between five and five and a"
        " quarter ounces.",
        "The fencing strip is fourteen meters long and one point"
        " five meters wide.",
        "A regulation discus for men weighs two kilograms and"
        " measures twenty-two centimeters across.",
        "The penalty spot in soccer is located twelve yards from"
        " the goal line.",
        "A squash court is thirty-two feet long and twenty-one"
        " feet wide.",
    ],
    # ------------------------------------------------------------------
    # GARDENING
    # ------------------------------------------------------------------
    NeutralDomain.GARDENING: [
        "Tomato seedlings need at least six hours of direct sunlight"
        " per day.",
        "Clay soil retains moisture longer than sandy soil in the"
        " same conditions.",
        "The raised garden bed measures four feet wide and eight"
        " feet long.",
        "Marigold seeds typically germinate within five to seven days"
        " at room temperature.",
        "A standard bag of potting mix weighs approximately forty"
        " pounds.",
        "Lavender plants prefer well-drained soil with a pH between"
        " six and eight.",
        "The garden hose delivers about nine gallons of water per"
        " minute at full pressure.",
        "Sunflower stems can grow to a height of twelve feet in a"
        " single growing season.",
        "Mulch applied at a depth of three inches helps retain soil"
        " moisture.",
        "The compost bin reached an internal temperature of one hundred"
        " forty degrees Fahrenheit.",
        "Basil leaves can be harvested once the plant reaches six"
        " inches tall.",
        "A drip irrigation emitter delivers water at a rate of one"
        " gallon per hour.",
        "Daffodil bulbs are typically planted at a depth of about six"
        " inches in autumn.",
        "The wheelbarrow holds approximately three cubic feet of"
        " soil per load.",
        "Zucchini plants produce both male and female flowers on"
        " the same vine.",
        "A soil pH meter measures acidity on a scale from zero to"
        " fourteen.",
        "Rosemary cuttings root best when placed in moist perlite"
        " for three weeks.",
        "The greenhouse thermostat was set to maintain sixty-five"
        " degrees Fahrenheit overnight.",
        "Pea plants use tendrils to climb vertical supports and"
        " trellises.",
        "The lawn mower blade spins at roughly three thousand"
        " revolutions per minute.",
        "Carrot seeds are sown at a depth of one quarter inch"
        " in loose soil.",
        "A standard terracotta pot has a drainage hole at the"
        " bottom center.",
        "Hostas grow best in partial shade with consistent moisture"
        " throughout the season.",
        "The pruning shears cut branches up to three quarters of"
        " an inch in diameter.",
        "Strawberry runners extend horizontally and root at the"
        " nodes to form new plants.",
        "A cubic yard of topsoil weighs approximately two thousand"
        " pounds.",
        "Pepper seedlings are transplanted outdoors after the"
        " last frost date passes.",
        "The rain barrel at the corner of the house collects fifty"
        " gallons from the downspout.",
        "Mint spreads rapidly through underground rhizomes in"
        " garden beds.",
        "The trellis stands six feet tall and supports climbing"
        " bean vines.",
    ],
    # ------------------------------------------------------------------
    # TRAVEL
    # ------------------------------------------------------------------
    NeutralDomain.TRAVEL: [
        "The direct flight from New York to London takes approximately"
        " seven hours.",
        "A standard carry-on suitcase measures twenty-two by fourteen"
        " by nine inches.",
        "The train from Tokyo to Osaka covers three hundred miles"
        " in about two and a half hours.",
        "Gate assignments at the airport terminal change on the"
        " departure board every few minutes.",
        "The hotel lobby had a seating area with four armchairs"
        " and a low table.",
        "A single-ride metro ticket in Paris costs two euros and"
        " fifteen cents.",
        "The ferry crossing between Dover and Calais takes roughly"
        " ninety minutes.",
        "Passport processing times average six to eight weeks for"
        " standard applications.",
        "The rental car odometer read forty-two thousand miles at"
        " pickup.",
        "An overnight sleeper train compartment typically fits two"
        " to four passengers.",
        "The airport terminal has three concourses connected by"
        " an underground shuttle.",
        "Checked luggage allowance on most transatlantic flights"
        " is fifty pounds per bag.",
        "The highway rest stop had fuel pumps, restrooms, and a"
        " small convenience store.",
        "A round-trip bus ticket from Boston to Philadelphia costs"
        " about thirty-five dollars.",
        "The cruise ship carried twenty-two hundred passengers and"
        " had fourteen decks.",
        "The time zone difference between Los Angeles and Chicago"
        " is two hours.",
        "A standard hotel room keycard uses radio-frequency"
        " identification technology.",
        "The cable car in San Francisco travels at a fixed speed"
        " of nine and a half miles per hour.",
        "Boarding passes display the gate number, seat assignment,"
        " and departure time.",
        "The highway toll between exits twelve and fifteen costs"
        " three dollars and seventy-five cents.",
        "A double-decker tour bus seats approximately seventy"
        " passengers on both levels.",
        "The luggage carousel in baggage claim is two hundred"
        " feet in total loop length.",
        "International flights typically begin boarding forty-five"
        " minutes before departure.",
        "The taxi meter started at three dollars and fifty cents"
        " with increments per quarter mile.",
        "A window seat on the left side of the plane faces north"
        " on eastbound flights.",
        "The hotel elevator serves floors one through eighteen"
        " and takes forty seconds to reach the top.",
        "A nonstop flight from Chicago to Denver covers about"
        " nine hundred miles.",
        "The train platform is four hundred meters long and"
        " serves twelve-car formations.",
        "The airport currency exchange booth posts daily rates"
        " on a digital display.",
        "A regional commuter rail pass covers all stations within"
        " a fifty-mile radius.",
    ],
    # ------------------------------------------------------------------
    # OFFICE
    # ------------------------------------------------------------------
    NeutralDomain.OFFICE: [
        "The laser printer produces approximately thirty pages"
        " per minute in draft mode.",
        "A standard letter-size sheet of paper measures eight and"
        " a half by eleven inches.",
        "The office supply cabinet contains three boxes of ballpoint"
        " pens and two reams of paper.",
        "A single ink cartridge yields about two hundred fifty"
        " printed pages.",
        "The adjustable desk raises from twenty-eight to forty-eight"
        " inches in height.",
        "The conference room table seats twelve people with chairs"
        " evenly spaced around it.",
        "A binder clip holds up to one hundred sheets of standard"
        " weight paper.",
        "The fluorescent ceiling lights operate at a color temperature"
        " of four thousand Kelvin.",
        "A standard filing cabinet has four drawers and stands"
        " fifty-two inches tall.",
        "The whiteboard in the meeting room measures six feet wide"
        " by four feet tall.",
        "A box of one thousand staples weighs approximately three"
        " and a half ounces.",
        "The desktop monitor has a twenty-seven-inch diagonal screen"
        " with a resolution of 2560 by 1440.",
        "A ream of printer paper contains five hundred sheets.",
        "The ergonomic keyboard has a split design with a twelve-degree"
        " tilt angle.",
        "The office thermostat was set to seventy-two degrees"
        " Fahrenheit.",
        "A standard paper clip is one and a quarter inches long"
        " and made of steel wire.",
        "The overhead projector displays at a native resolution"
        " of 1920 by 1080 pixels.",
        "A gel ink pen has an average writing length of one"
        " thousand meters.",
        "The three-hole punch aligns holes at standard spacing"
        " of four and a quarter inches apart.",
        "The label maker prints adhesive strips up to half an"
        " inch wide.",
        "A mechanical pencil uses zero point five millimeter lead"
        " refills.",
        "The office bookshelf holds approximately eighty binders"
        " across five shelves.",
        "A sticky note pad contains one hundred individual sheets"
        " in a three-by-three-inch size.",
        "The shredder processes up to ten sheets of paper at a"
        " time in crosscut mode.",
        "The desk lamp uses a nine-watt LED bulb rated at eight"
        " hundred lumens.",
        "A standard manila folder measures nine and a half by"
        " eleven and three-quarter inches.",
        "The scanner captures documents at a resolution of six"
        " hundred dots per inch.",
        "An electric pencil sharpener sharpens a standard pencil"
        " in about three seconds.",
        "The cork bulletin board measures three feet by two feet"
        " and hangs on two hooks.",
        "A roll of packing tape is one hundred ten yards long"
        " and two inches wide.",
    ],
    # ------------------------------------------------------------------
    # MUSIC
    # ------------------------------------------------------------------
    NeutralDomain.MUSIC: [
        "A standard piano has eighty-eight keys spanning seven"
        " full octaves plus three extra notes.",
        "The A above middle C vibrates at a frequency of four"
        " hundred forty hertz.",
        "A set of guitar strings includes six strings of varying"
        " gauge and material.",
        "The metronome was set to one hundred twenty beats per"
        " minute for the rehearsal.",
        "A concert grand piano measures approximately nine feet"
        " in length.",
        "The snare drum head is typically fourteen inches in"
        " diameter.",
        "A violin bow contains between one hundred fifty and two"
        " hundred horsehairs.",
        "The soprano saxophone is the third smallest member of"
        " the saxophone family.",
        "A standard set of orchestral timpani includes four"
        " drums of different sizes.",
        "The trombone extends its slide to seven positions to"
        " change pitch.",
        "A twelve-string guitar has six pairs of strings tuned"
        " in octaves and unisons.",
        "The bass clarinet sounds one octave lower than the"
        " standard B-flat clarinet.",
        "A cello has four strings tuned to C, G, D, and A from"
        " lowest to highest.",
        "The pipe organ in the hall has three manuals and a"
        " pedalboard with thirty-two keys.",
        "A standard drumstick measures sixteen inches in length"
        " and weighs about two ounces.",
        "The electric bass guitar typically has four strings"
        " tuned E, A, D, and G.",
        "A ukulele has four nylon strings and a scale length"
        " of about thirteen inches.",
        "The French horn has approximately twelve feet of coiled"
        " brass tubing.",
        "A clarinet reed is made from a single piece of cane"
        " about three inches long.",
        "The sustain pedal on a piano lifts all the dampers"
        " off the strings simultaneously.",
        "A piccolo plays one octave higher than a standard"
        " concert flute.",
        "The double bass stands approximately six feet tall"
        " when resting on its endpin.",
        "An oboe uses a double reed made from two thin pieces"
        " of cane tied together.",
        "The hi-hat cymbal stand has a foot pedal that opens"
        " and closes the two cymbals.",
        "A banjo has a round body with a stretched membrane"
        " that acts as the soundboard.",
        "The xylophone bars are arranged in two rows similar to"
        " a piano keyboard layout.",
        "A trumpet has three piston valves that redirect air"
        " through additional tubing.",
        "The harp has forty-seven strings and seven foot pedals"
        " for changing key.",
        "A music stand adjusts in height from twenty-eight to"
        " fifty inches.",
        "The tuning peg on a guitar rotates to increase or decrease"
        " string tension.",
    ],
    # ------------------------------------------------------------------
    # CONSTRUCTION
    # ------------------------------------------------------------------
    NeutralDomain.CONSTRUCTION: [
        "A standard two-by-four lumber piece actually measures one"
        " and a half by three and a half inches.",
        "Portland cement hardens through a chemical reaction with"
        " water called hydration.",
        "The concrete foundation was poured at a thickness of"
        " eight inches.",
        "A sheet of half-inch plywood weighs approximately forty"
        " to forty-eight pounds.",
        "Rebar is placed inside concrete forms to increase tensile"
        " strength.",
        "The steel I-beam spans thirty feet across the building"
        " opening.",
        "A standard brick measures approximately eight by four"
        " by two and a quarter inches.",
        "The roof truss is constructed with two-by-six lumber"
        " and metal gusset plates.",
        "A bag of ready-mix concrete weighs eighty pounds and"
        " yields about six tenths of a cubic foot.",
        "The circular saw blade spins at five thousand revolutions"
        " per minute.",
        "Drywall sheets come in standard dimensions of four"
        " feet by eight feet.",
        "The tape measure extends to twenty-five feet and has"
        " markings in sixteenths of an inch.",
        "Galvanized nails resist corrosion better than uncoated"
        " steel nails.",
        "The spirit level showed a bubble centered between the"
        " two reference lines.",
        "A cubic yard of wet concrete weighs approximately four"
        " thousand pounds.",
        "The pneumatic nail gun drives framing nails three and"
        " a half inches long.",
        "Insulation batts for standard walls are three and a"
        " half inches thick.",
        "The scaffold platform stands twelve feet above ground"
        " level on adjustable legs.",
        "A five-gallon bucket of joint compound weighs about"
        " sixty-two pounds.",
        "The mason laid twelve courses of brick to reach the"
        " window sill height.",
        "Copper pipes used in residential plumbing are typically"
        " half-inch or three-quarter-inch diameter.",
        "The concrete mixer drum rotates at approximately fifteen"
        " revolutions per minute during mixing.",
        "Oriented strand board is manufactured from compressed"
        " wood strands bonded with resin.",
        "The framing hammer has a twenty-ounce steel head and"
        " a sixteen-inch handle.",
        "A box of three-inch wood screws contains one hundred"
        " pieces.",
        "The excavator bucket has a capacity of one and a half"
        " cubic yards.",
        "Mortar mix consists of cement, lime, and sand in"
        " measured proportions.",
        "The laser distance measurer is accurate to within one"
        " sixteenth of an inch at one hundred feet.",
        "Pressure-treated lumber contains preservatives that"
        " resist insect damage and rot.",
        "The bulldozer blade is twelve feet wide and pushes"
        " soil at a depth of eighteen inches.",
    ],
    # ------------------------------------------------------------------
    # ASTRONOMY
    # ------------------------------------------------------------------
    NeutralDomain.ASTRONOMY: [
        "The distance from Earth to the Moon averages about two"
        " hundred thirty-nine thousand miles.",
        "Jupiter completes one full rotation on its axis in"
        " approximately ten hours.",
        "Light from the Sun takes about eight minutes and twenty"
        " seconds to reach Earth.",
        "The Andromeda Galaxy is approximately two point five"
        " million light-years from the Milky Way.",
        "Saturn's rings are composed primarily of ice particles"
        " and rocky debris.",
        "A solar eclipse occurs when the Moon passes directly"
        " between the Earth and the Sun.",
        "The Hubble Space Telescope orbits Earth at an altitude"
        " of about three hundred forty miles.",
        "Mars has two small moons named Phobos and Deimos.",
        "The surface temperature of the Sun is approximately"
        " ten thousand degrees Fahrenheit.",
        "Venus rotates in the opposite direction compared to"
        " most other planets in the solar system.",
        "The International Space Station orbits Earth roughly"
        " every ninety minutes.",
        "Neptune takes about one hundred sixty-five Earth years"
        " to complete one orbit around the Sun.",
        "A light-year equals approximately five point eight"
        " eight trillion miles.",
        "The asteroid belt is located between the orbits of"
        " Mars and Jupiter.",
        "Mercury's surface temperature ranges from negative"
        " two hundred eighty to eight hundred degrees Fahrenheit.",
        "The Milky Way galaxy contains an estimated one hundred"
        " to four hundred billion stars.",
        "A full Moon occurs approximately every twenty-nine"
        " and a half days.",
        "The Great Red Spot on Jupiter is a storm larger than"
        " the diameter of Earth.",
        "Proxima Centauri is the nearest star to the Sun at"
        " four point two four light-years away.",
        "The Orion Nebula is visible to the naked eye and"
        " lies about thirteen hundred light-years from Earth.",
        "Pluto's orbital period around the Sun is approximately"
        " two hundred forty-eight Earth years.",
        "The Voyager 1 spacecraft is currently over fourteen"
        " billion miles from Earth.",
        "Titan, Saturn's largest moon, has a thick nitrogen"
        " atmosphere and surface lakes of methane.",
        "A neutron star has a diameter of only about twelve"
        " miles despite enormous mass.",
        "The Kuiper Belt extends from Neptune's orbit outward"
        " to about fifty astronomical units.",
        "The refracting telescope uses two glass lenses to"
        " magnify distant objects.",
        "Earth's axial tilt of twenty-three point five degrees"
        " causes the cycle of seasons.",
        "The Crab Nebula is the remnant of a supernova observed"
        " in the year 1054.",
        "Io, one of Jupiter's moons, has over four hundred"
        " active volcanoes on its surface.",
        "The observable universe has a diameter of approximately"
        " ninety-three billion light-years.",
    ],
    # ------------------------------------------------------------------
    # TEXTILES
    # ------------------------------------------------------------------
    NeutralDomain.TEXTILES: [
        "Cotton fibers grow in bolls on the cotton plant and"
        " are harvested once mature.",
        "A standard bolt of fabric is typically between forty"
        " and one hundred yards long.",
        "Polyester fabric dries faster than cotton fabric of"
        " the same thickness.",
        "The loom weaves horizontal weft threads through vertical"
        " warp threads.",
        "Silk fibers are produced by silkworms and can be up"
        " to one mile long per cocoon.",
        "Denim is a twill-weave fabric made from cotton yarn"
        " dyed with indigo.",
        "A sewing machine bobbin holds approximately seventy"
        " yards of thread.",
        "Merino wool fibers measure between seventeen and"
        " twenty-five microns in diameter.",
        "The fabric bolt on the shelf contains sixty yards"
        " of medium-weight linen.",
        "Thread count refers to the number of threads per"
        " square inch in woven fabric.",
        "Nylon was first commercially produced in 1939 as"
        " a synthetic alternative to silk.",
        "A serger machine trims and overcasts fabric edges"
        " simultaneously using multiple threads.",
        "Flannel fabric has a soft napped surface created"
        " by brushing the woven fibers.",
        "The spinning wheel twists raw fiber into yarn at"
        " a rate set by the drive ratio.",
        "A yard of canvas fabric weighs approximately ten"
        " to twelve ounces.",
        "Rayon is manufactured from dissolved wood cellulose"
        " that is extruded into fibers.",
        "The rotary cutter slices through four layers of"
        " quilting cotton in a single pass.",
        "Tweed is a rough-textured wool fabric originally"
        " woven in Scotland.",
        "A standard sewing needle ranges from size nine"
        " to size eighteen for common fabrics.",
        "Velvet has a dense pile surface created by cutting"
        " loops in the weave.",
        "The industrial loom produces approximately sixty"
        " yards of fabric per hour.",
        "Linen fabric is woven from fibers of the flax"
        " plant and wrinkles easily.",
        "A thimble protects the fingertip when pushing"
        " a needle through heavy fabric.",
        "Cashmere fibers come from the undercoat of"
        " cashmere goats and are very fine.",
        "The selvage is the tightly woven edge that runs"
        " along both sides of a bolt of fabric.",
        "Satin weave creates a smooth glossy surface by"
        " floating warp threads over several weft threads.",
        "A pincushion holds between thirty and fifty"
        " straight pins in a compact form.",
        "Acrylic yarn is a synthetic fiber often used as"
        " a lower-cost substitute for wool.",
        "The buttonhole attachment on the sewing machine"
        " stitches a one-inch opening in four steps.",
        "Organza is a thin, sheer fabric woven from"
        " continuous filament silk or synthetic fibers.",
    ],
}


def get_flat_neutral_pool() -> list[tuple[str, NeutralDomain]]:
    """Return all neutral sentences as a flat list of (sentence, domain) tuples.

    Returns:
        List of tuples where each tuple contains the sentence text and its
        corresponding NeutralDomain enum value.
    """
    result: list[tuple[str, NeutralDomain]] = []
    for domain, sentences in NEUTRAL_POOL.items():
        for sentence in sentences:
            result.append((sentence, domain))
    return result

"""
Dataset v2: same output grammar as v1, but far more paraphrases per intent
(the "teacher" role in distillation), typo / word-dropout augmentation and
copy-training for free-text slots (timer labels, LED colors).

Fairness rules, enforced by checks below:
  * no training text may equal any text in test_novel or test_real
  * BANNED words (the distinctive held-out words of test_novel) never appear in training
"""
import json
import os
import random
import string
CHARS = set(" abcdefghijklmnopqrstuvwxyz0123456789'.,?!%-=|+_")


def normalize(text):
    """Lowercase, drop characters outside the fixed alphabet, collapse spaces."""
    return " ".join("".join(c for c in text.lower() if c in CHARS).split())

BANNED = {"flip", "lighting", "lamps", "cut", "cooler", "going", "output", "activate", "deactivate",
          "cyan", "magenta", "paint", "ease", "firmer", "seize", "clamp", "park", "coffee", "bread",
          "wake", "ping", "moisture", "detect", "outside", "measure", "leave", "spinning", "kindly",
          "yo", "hallway", "study", "lounge", "peru", "norway", "emma", "volcanoes", "telephone",
          "greek", "lasagna", "invented", "recipe", "draft", "couple", "quarter", "passed", "wait",
          "seized"}

ROOMS = {"kitchen": "kitchen", "bedroom": "bedroom", "bed room": "bedroom", "living room": "living_room",
         "livingroom": "living_room", "bathroom": "bathroom", "bath room": "bathroom", "hall": "hall",
         "office": "office", "garage": "garage"}
COLORS = ["red", "green", "blue", "white", "yellow", "purple", "orange", "pink", "violet", "gold", "teal",
          "lime", "amber", "brown", "silver", "indigo", "crimson", "maroon", "navy", "turquoise", "lavender",
          "beige", "rose", "mint", "coral", "olive", "peach", "azure", "black", "grey", "gray"]
LABELS = ["tea", "pasta", "eggs", "laundry", "pizza", "rice", "nap", "break", "cake", "soup", "oven",
          "meeting", "workout", "bus", "call", "homework", "medicine", "steak", "cookies", "dishes",
          "plants", "game", "potatoes", "noodles", "yoga", "reading", "class", "train", "beans", "fish"]
ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen " \
       "sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def num_word(n):
    if n == 100:
        return "one hundred"
    if n < 20:
        return ONES[n]
    t, o = divmod(n, 10)
    return TENS[t] + ("" if o == 0 else rng_choice([" ", "-"]) + ONES[o])


_R = random.Random(0)


def rng_choice(x):
    return _R.choice(x)


def ntext(rng, n, p_word=0.35):
    return num_word(n).replace("-", " ") if rng.random() < p_word else str(n)


def rand_word(rng):
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3, 8)))


# ---------------- durations ----------------
UNIT_PL = {"s": ["seconds", "secs", "sec"], "m": ["minutes", "mins", "min"], "h": ["hours", "hrs", "hr"]}
UNIT_SG = {"s": ["second", "sec"], "m": ["minute", "min"], "h": ["hour", "hr"]}
SPECIAL = [("half an hour", "30m"), ("an hour", "1h"), ("a minute", "1m"), ("a second", "1s"),
           ("one hour", "1h"), ("an hour and a half", "90m"), ("one minute", "1m"), ("half a minute", "30s")]


def dur(rng, adjective=False):
    """(text, canonical). adjective=True gives '5 minute' as in 'a 5 minute timer'."""
    if not adjective and rng.random() < 0.12:
        return rng.choice(SPECIAL)
    u = rng.choice("smmmhh")
    n = {"s": rng.choice([5, 10, 15, 20, 25, 30, 40, 45, 50, 60, 90, rng.randint(1, 99)]),
         "m": rng.randint(1, 99), "h": rng.randint(1, 24)}[u]
    unit = rng.choice(UNIT_SG[u] if (adjective or n == 1) else UNIT_PL[u])
    return f"{ntext(rng, n)} {unit}", f"{n}{u}"


def delay(rng):
    d, c = dur(rng)
    tpl = rng.choice(["in {d}", "after {d}", "{d} from now", "{d} later", "in about {d}", "after about {d}",
                      "in roughly {d}", "in exactly {d}", "after {d} or so"])
    return tpl.format(d=d), c


def room(rng, p=0.4):
    if rng.random() < p:
        k = rng.choice(list(ROOMS))
        return k, ROOMS[k]
    return None, None


# ---------------- intents ----------------
LIGHT_ON = ["turn on the {L}", "turn the {L} on", "switch on the {L}", "switch the {L} on", "{L} on",
            "put the {L} on", "enable the {L}", "power on the {L}", "power up the {L}", "i need the {L} on",
            "get the {L} on", "{L} on please", "start the {L}", "open the {L}", "bring the {L} on",
            "i want the {L} on", "could you turn on the {L}", "set the {L} on", "set {L} to on", "{L} to on",
            "the {L} should be on", "on with the {L}", "it's dark, {L} on", "it is too dark, turn on the {L}",
            "too dark in here, switch on the {L}", "make the {L} come on", "light on in here",
            "can i have the {L} on", "i can't see, turn on the {L}", "{L} on now"]
LIGHT_OFF = ["turn off the {L}", "turn the {L} off", "switch off the {L}", "switch the {L} off", "{L} off",
             "{L} out", "put the {L} out", "put out the {L}", "disable the {L}", "power off the {L}",
             "power down the {L}", "kill the {L}", "shut off the {L}", "shut the {L} off", "close the {L}",
             "i don't need the {L}", "no more {L}", "{L} to off", "set {L} to off", "stop the {L}",
             "it's too bright, turn off the {L}", "time to sleep, {L} off", "time for bed, {L} off",
             "the {L} should be off", "get the {L} off", "off with the {L}", "{L} off please",
             "turn every {L} off", "switch all {L} off", "darken the room"]
LIGHT_NOUN = ["light", "lights", "lamp", "ceiling light", "main light", "overhead light", "lights", "light"]


def g_light(rng):
    on = rng.random() < 0.5
    tpl = rng.choice(LIGHT_ON if on else LIGHT_OFF)
    L = rng.choice(LIGHT_NOUN)
    if rng.random() < 0.1:
        L = rng.choice(["all the lights", "all lights", "every light"])
    rt, r = room(rng)
    if rt and "{L}" in tpl:
        L = rng.choice([f"{rt} {L}", f"{L} in the {rt}", f"{L} in {rt}"])
    elif rt:
        rt, r = None, None
    text = tpl.format(L=L).replace("the all", "all").replace("every every", "every")
    return text, f"light {'on' if on else 'off'}" + (f" room={r}" if r else "")


FAN_N = ["fan", "fan", "ventilator", "ceiling fan", "desk fan", "fan"]
FAN_ON = ["turn on the {F}", "start the {F}", "{F} on", "switch the {F} on", "switch on the {F}", "run the {F}",
          "power on the {F}", "enable the {F}", "get the {F} running", "i need the {F}", "it's hot, turn the {F} on",
          "too hot in here, start the {F}", "{F} on please", "turn the {F} on", "fire up the {F}",
          "start up the {F}", "set the {F} on", "i'm sweating, {F} on", "give me some air", "blow some air"]
FAN_OFF = ["turn off the {F}", "stop the {F}", "{F} off", "switch the {F} off", "switch off the {F}",
           "disable the {F}", "power off the {F}", "kill the {F}", "turn the {F} off", "{F} off please",
           "i'm cold, stop the {F}", "enough {F}", "no more {F}", "halt the {F}", "shut off the {F}",
           "stop the {F} from running", "set the {F} off", "{F} stop"]
SPEED_W = {1: ["low", "slow", "minimum", "lowest", "gentle", "min"],
           2: ["medium", "middle", "normal", "half", "moderate"],
           3: ["high", "max", "full", "maximum", "highest", "full power", "full blast"]}
SPEED_TPL = ["set the {F} to {v}", "{F} on {v}", "{F} to {v}", "run the {F} at {v}", "{F} {v}",
             "turn the {F} to {v}", "switch the {F} to {v}", "change the {F} to {v}", "{F} at {v} speed",
             "{F} on {v} speed", "set {F} speed {v}", "{v} fan", "{F} at {v}", "put {F} to {v}",
             "set {F} to {v} speed", "run the {F} on {v}"]
SPEED_N = ["set {F} speed to {n}", "{F} speed {n}", "{F} to speed {n}", "speed {n} on the {F}", "{F} level {n}",
           "set the {F} to level {n}", "change the {F} speed to {n}", "{F} speed level {n}",
           "switch the {F} to speed {n}", "slow the {F} to speed {n}", "speed up the {F} to {n}"]


def g_fan(rng):
    F = rng.choice(FAN_N)
    r = rng.random()
    if r < 0.33:
        return rng.choice(FAN_ON).format(F=F), "fan on"
    if r < 0.66:
        return rng.choice(FAN_OFF).format(F=F), "fan off"
    s = rng.randint(1, 3)
    if rng.random() < 0.55:
        return rng.choice(SPEED_TPL).format(F=F, v=rng.choice(SPEED_W[s])), f"fan speed={s}"
    return rng.choice(SPEED_N).format(F=F, n=ntext(rng, s, 0.5)), f"fan speed={s}"


REL_N = ["relay", "switch", "channel", "port", "socket", "relay", "plug", "relay number", "switch number"]
REL_ON = ["turn on {R} {n}", "{R} {n} on", "enable {R} {n}", "close {R} {n}", "switch on {R} {n}",
          "power on {R} {n}", "turn {R} {n} on", "set {R} {n} on", "set {R} {n} to on", "energize {R} {n}",
          "start {R} {n}", "{R} {n} to on", "switch {R} {n} on", "{R} {n} on please", "power up {R} {n}",
          "engage {R} {n}"]
REL_OFF = ["turn off {R} {n}", "{R} {n} off", "disable {R} {n}", "open {R} {n}", "switch off {R} {n}",
           "power off {R} {n}", "turn {R} {n} off", "set {R} {n} off", "set {R} {n} to off", "kill {R} {n}",
           "stop {R} {n}", "{R} {n} to off", "switch {R} {n} off", "release {R} {n}", "power down {R} {n}",
           "disengage {R} {n}"]


def g_relay(rng):
    n = rng.randint(1, 8)
    on = rng.random() < 0.5
    t = rng.choice(REL_ON if on else REL_OFF).format(R=rng.choice(REL_N), n=ntext(rng, n, 0.4))
    return t, f"relay {n} {'on' if on else 'off'}"


LED_TPL = ["make the led {c}", "set led to {c}", "led {c}", "change the led color to {c}", "turn the led {c}",
           "set the led color {c}", "led color {c}", "switch the led to {c}", "{c} led", "{c} led please",
           "show {c} on the led", "light the led {c}", "i want the led {c}", "can the led be {c}",
           "change led to {c}", "led to {c}", "set led color to {c}", "led should be {c}", "go {c} on the led",
           "the led {c} please", "color the led {c}", "use {c} for the led", "set the led to {c} color"]
LED_OFF = ["led off", "turn off the led", "switch the led off", "disable the led", "led out", "no led",
           "turn the led off", "power off the led", "stop the led", "switch off the led", "kill the led light"]


def g_led(rng):
    if rng.random() < 0.18:
        return rng.choice(LED_OFF), "led off"
    c = rand_word(rng) if rng.random() < 0.25 else rng.choice(COLORS)
    return rng.choice(LED_TPL).format(c=c), f"led {c}"


SOFTER = ["grip softer", "loosen the grip", "squeeze less", "less force", "grip a bit softer", "softer",
          "gentler", "be gentle", "gentler grip", "lighter grip", "relax the grip", "reduce grip force",
          "less pressure", "loosen it", "loosen up", "too tight", "that's too hard", "decrease the grip",
          "lower the grip force", "grip less", "not too hard", "a bit softer", "you're squeezing too hard",
          "careful, softer", "less tight", "lighten the grip", "you'll crush it, softer", "reduce the pressure"]
HARDER = ["grip harder", "tighten the grip", "squeeze more", "more force", "grip a bit harder", "harder",
          "stronger", "stronger grip", "increase grip force", "more pressure", "tighten it", "tighten up",
          "too loose", "it's slipping", "grip more", "raise the grip force", "squeeze tighter",
          "a bit harder", "it will fall, grip harder", "hold it more firmly", "tighter",
          "increase the pressure", "don't drop it, squeeze more", "more tight", "add more force"]
FORCE = ["grip with {f} percent force", "grip at {f}%", "set grip force to {f}", "squeeze at {f} percent",
         "grip force {f}", "{f} percent grip", "grip {f}%", "use {f}% force", "grip at {f} percent",
         "set the grip to {f}", "squeeze with {f}% force", "apply {f} percent force", "force {f}",
         "grip force to {f} percent", "set force {f}%", "squeeze {f} percent", "grip strength {f}",
         "{f}% grip force"]
GRAB = ["grab it", "close the gripper", "grip", "pick it up", "grab the object", "grab", "take it",
        "pick up the item", "catch it", "hold it", "hold the object", "grasp it", "grasp", "close the hand",
        "close gripper", "take the object", "pick that up", "get it", "grab the thing", "grab the item",
        "close your hand", "take hold of it", "pick up that thing", "grip the object", "hold that"]


def g_grip(rng):
    r = rng.random()
    if r < 0.25:
        return rng.choice(SOFTER), "grip adjust=-20"
    if r < 0.5:
        return rng.choice(HARDER), "grip adjust=+20"
    if r < 0.75:
        f = rng.choice([rng.randint(0, 100), rng.randrange(0, 101, 5), rng.randrange(0, 101, 10)])
        return rng.choice(FORCE).format(f=ntext(rng, f, 0.15)), f"grip force={f}"
    return rng.choice(GRAB), "grip"


RELEASE = ["release", "open the gripper", "let go", "drop it", "release the object", "open the claw",
           "open the hand", "let it drop", "release it", "drop", "open gripper", "free it", "let the object go",
           "release grip", "drop the item", "ungrip", "open your hand", "let go of the object", "release the item",
           "drop the thing", "open up the claw", "loosen completely", "unhand it", "set it free", "let that go"]


def g_release(rng):
    return rng.choice(RELEASE), "release"


HOME = ["arm home", "go home", "return the arm home", "reset the arm", "move arm to home", "home position",
        "go to home position", "return to home", "back home", "arm to home", "home the arm", "go back home",
        "reset position", "return arm", "move home", "home", "send the arm back home", "arm back to home",
        "return to the home position", "move the arm home", "go to the home spot", "rest position"]
CELL = ["move to cell {c}", "go to cell {c}", "arm to cell {c}", "move the arm to slot {c}", "reach cell {c}",
        "cell {c}", "slot {c}", "go to slot {c}", "move to slot {c}", "move arm to position {c}",
        "go to spot {c}", "move over to cell {c}", "arm cell {c}", "travel to cell {c}", "point at cell {c}",
        "goto cell {c}", "move the arm over cell {c}", "take the arm to cell {c}", "position the arm at cell {c}",
        "go over to slot {c}", "move it to cell {c}", "bring the arm to slot {c}", "cell number {c}"]


def g_arm(rng):
    if rng.random() < 0.3:
        return rng.choice(HOME), "arm home"
    c = rng.randint(1, 9)
    return rng.choice(CELL).format(c=ntext(rng, c, 0.4)), f"arm cell={c}"


TIMER_L = ["set a {a} timer for {l}", "{l} timer {d}", "timer for {l} {d}", "remind me about the {l} in {d}",
           "{l} in {d}", "timer {d} for {l}", "{a} timer for the {l}", "set a timer for the {l} for {d}",
           "{l} timer for {d}", "set {l} timer to {d}", "timer for the {l}, {d}", "{a} {l} timer please",
           "check the {l} in {d}", "remind me of the {l} in {d}", "the {l} needs {d}, set a timer"]
TIMER = ["set a timer for {d}", "timer {d}", "start a {a} timer", "countdown {d}", "{a} timer", "set timer {d}",
         "timer for {d}", "count down {d}", "start a countdown for {d}", "remind me in {d}", "alert me in {d}",
         "let me know in {d}", "notify me in {d}", "beep in {d}", "set a {a} countdown", "give me {d}",
         "timer of {d}", "i need a {a} timer", "make a timer for {d}", "start timer {d}"]


def g_timer(rng):
    d, can = dur(rng)
    a, can_a = dur(rng, adjective=True)
    if rng.random() < 0.4:
        lab = rand_word(rng) if rng.random() < 0.3 else rng.choice(LABELS)
        tpl = rng.choice(TIMER_L)
    else:
        lab, tpl = None, rng.choice(TIMER)
    if "{a}" in tpl:
        d, can = a, can_a
    return tpl.format(d=d, a=a, l=lab), f"timer {can}" + (f" label={lab}" if lab else "")


SENS = {"temp": ["what is the temperature", "what's the temperature", "how hot is it", "read the temperature",
                 "temperature", "how warm is it", "temp", "current temperature", "temperature reading",
                 "how cold is it", "is it hot", "what temp is it", "tell me the temperature",
                 "get temperature", "check temperature", "how many degrees is it", "degrees",
                 "how hot is it right now", "temperature please", "what's the temp"],
        "humidity": ["what is the humidity", "how humid is it", "read humidity", "humidity level", "humidity",
                     "check humidity", "air humidity", "how wet is the air", "is it humid", "get humidity",
                     "humidity reading", "tell me the humidity", "what's the humidity", "humidity now",
                     "how much humidity is there", "relative humidity"],
        "motion": ["is there any motion", "did someone move", "check motion sensor", "any movement", "motion",
                   "motion sensor", "is someone here", "anyone moving", "any motion", "check for movement",
                   "is somebody in the room", "motion reading", "has anything moved", "read the motion sensor",
                   "is anybody moving", "check the motion"],
        "light": ["how bright is it", "read the light sensor", "light level", "what is the brightness",
                  "brightness", "brightness level", "how dark is it", "ambient light", "light sensor",
                  "check brightness", "is it bright", "light sensor reading", "what's the light level",
                  "how much light is there", "read brightness"]}


def g_sensor(rng):
    k = rng.choice(list(SENS))
    return rng.choice(SENS[k]), f"sensor {k}"


ST_LIGHT = ["is the {L} on", "are the {L} on", "check the {L}", "status of the {L}", "are the {L} off",
            "is the {L} off", "is the {L} still on", "{L} status", "are the {L} still on", "check if the {L} is on",
            "tell me if the {L} is on", "what's the state of the {L}"]
ST = {"fan": ["is the fan running", "fan status", "is the fan on", "is the fan off", "is the fan working",
              "check the fan", "is the fan still running", "fan state", "what's the fan doing"],
      "arm": ["where is the arm", "arm status", "arm position", "where is the arm now", "check the arm",
              "arm location", "where's the arm right now", "what position is the arm in", "arm state"],
      "": ["status", "system status", "how are you doing", "report status", "status report", "are you ok",
           "health check", "diagnostics", "how is everything", "are you alive", "are you working",
           "run diagnostics", "self check", "device status", "how's it going"]}


def g_status(rng):
    r = rng.random()
    if r < 0.4:
        L = rng.choice(["light", "lights", "lamp"])
        t = rng.choice(ST_LIGHT).format(L=L)
        rt, rm = room(rng, 0.5)
        if rt:
            t += rng.choice([f" in the {rt}", f" in {rt}"])
        return t, "status light" + (f" room={rm}" if rm else "")
    k = "fan" if r < 0.6 else ("arm" if r < 0.8 else "")
    return rng.choice(ST[k]), ("status " + k).strip()


CLOUD_TPL = ["what is the capital of {x}", "tell me a joke", "who wrote {b}", "how far is the moon",
             "write a poem about {t}", "what is the weather in {x}", "explain {t}", "translate {w} to {l}",
             "what is {a} plus {c}", "who is the president of {x}", "recommend a movie", "how do i cook {f}",
             "what time is it in {x}", "summarize the news", "tell me about {t}", "why is the sky blue",
             "book a flight to {x}", "send an email to my boss", "what should i eat today", "play some music",
             "what's {a} times {c}", "how old is the universe", "what is {t}", "teach me about {t}",
             "write a story about {t}", "what's the population of {x}", "who discovered {t}",
             "how many people live in {x}", "can you help me with my homework", "what's the news today",
             "tell me a story", "what is love", "how do i learn {l}", "what's the best {p}",
             "give me advice about {t}", "write a song about {t}", "what are you", "who made you",
             "what's {a} divided by {c}", "how tall is mount everest", "remind me what {t} is",
             "should i buy a {p}", "compare {x} and {y}", "what language do they speak in {x}",
             "plan a trip to {x}", "order food", "call {n}", "text {n} that i'm late", "open youtube",
             "search the web for {t}", "what's the meaning of {w}", "how do magnets work",
             "what's the distance to {x}", "is {x} a big country", "describe {t}", "tell me a riddle",
             "what's trending", "how do i fix my car", "write code for a website", "what is ai",
             "who are you", "what can you do", "i'm bored", "i feel sad", "good morning", "hello there",
             "thank you so much", "what do you think about {t}", "do you like {t}", "how's the weather"]
FILL = {"x": ["france", "japan", "brazil", "kenya", "canada", "italy", "india", "egypt", "spain", "chile",
              "mexico", "china", "germany", "poland", "vietnam", "ghana", "sweden", "turkey", "korea", "fiji"],
        "y": ["france", "japan", "brazil", "italy", "india", "spain", "mexico", "germany"],
        "b": ["hamlet", "dune", "the hobbit", "1984", "dracula", "ulysses", "frankenstein", "macbeth"],
        "t": ["black holes", "the ocean", "love", "rockets", "gravity", "dinosaurs", "electricity", "dna",
              "the roman empire", "bitcoin", "the internet", "photosynthesis", "jazz", "chess", "penguins",
              "the sun", "mars", "football", "democracy", "robots", "evolution", "the brain", "rainbows"],
        "l": ["french", "german", "spanish", "italian", "japanese", "chinese", "arabic", "polish"],
        "w": ["hello", "thank you", "good night", "cat", "friend", "water", "happiness"],
        "f": ["rice", "pancakes", "soup", "pasta", "steak", "curry", "omelette", "cookies", "fish"],
        "p": ["phone", "laptop", "car", "camera", "book", "movie", "game", "bike"],
        "n": ["mom", "dad", "john", "my sister", "alex", "my boss", "sarah"]}


def g_cloud(rng):
    if rng.random() < 0.08:
        n = rng.randint(3, 25)
        t = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz    ") for _ in range(n)).strip()
        return (t or "hmm"), "cloud"
    t = rng.choice(CLOUD_TPL)
    for k, v in FILL.items():
        while "{" + k + "}" in t:
            t = t.replace("{" + k + "}", rng.choice(v), 1)
    t = t.replace("{a}", str(rng.randint(1, 999))).replace("{c}", str(rng.randint(1, 99)))
    return t, "cloud"


SIMPLE = [(g_light, 3), (g_fan, 2), (g_relay, 2), (g_led, 1.5), (g_grip, 2.5), (g_release, 1),
          (g_arm, 2), (g_timer, 2), (g_sensor, 1.5), (g_status, 1.5), (g_cloud, 3)]
DELAYABLE = {g_light, g_fan, g_relay}
PREFIX = ["please ", "can you ", "hey ", "could you ", "would you ", "i want you to ", "hey robot ", "ok ",
          "now ", "go ahead and ", "i'd like you to ", "robot, ", "quickly ", "okay ", "alright, ", "pls "]
SUFFIX = [" please", " now", " thanks", " thank you", " right now", " asap", " for me", "!", ".", " quickly",
          " pls", " if you can", " ok"]
JOIN = [" and ", " then ", " and then ", ", ", " also ", ", then ", " plus ", ". ", " and afterwards ",
        ", and ", " then also "]


def one(rng):
    fns, ws = zip(*SIMPLE)
    fn = rng.choices(fns, ws)[0]
    text, tgt = fn(rng)
    if fn in DELAYABLE and rng.random() < 0.3:
        dt, dc = delay(rng)
        text = f"{dt} {text}" if rng.random() < 0.4 else f"{text} {dt}"
        tgt += f" delay={dc}"
    return text, tgt


KEYS = "qwertyuiop asdfghjkl zxcvbnm".split()


def neighbor(c, rng):
    for row in KEYS:
        i = row.find(c)
        if i >= 0:
            return row[max(0, min(len(row) - 1, i + rng.choice([-1, 1])))]
    return c


def augment(rng, text, tgt):
    """Typos and dropped filler words. Words copied into the target are protected."""
    protect = set(tgt.replace("|", " ").replace("=", " ").split())
    words = text.split()
    out = []
    for w in words:
        if w in ("the", "a", "an", "to") and rng.random() < 0.25:
            continue
        if len(w) > 3 and w.isalpha() and w not in protect and rng.random() < 0.12:
            i = rng.randrange(len(w) - 1)
            op = rng.random()
            if op < 0.3:
                w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
            elif op < 0.55:
                w = w[:i] + w[i + 1:]
            elif op < 0.8:
                w = w[:i] + w[i] + w[i:]
            else:
                w = w[:i] + neighbor(w[i], rng) + w[i + 1:]
        out.append(w)
    return " ".join(out) or text


def sample(rng):
    text, tgt = one(rng)
    if tgt != "cloud" and rng.random() < 0.2:
        t2, g2 = one(rng)
        if g2 != "cloud":
            text, tgt = text + rng.choice(JOIN) + t2, tgt + "|" + g2
    if rng.random() < 0.18:
        text = rng.choice(PREFIX) + text
    if rng.random() < 0.12:
        text += rng.choice(SUFFIX)
    if rng.random() < 0.25:
        text = augment(rng, text, tgt)
    return normalize(text), tgt


def load_tests():
    forbid = set()
    for l in open("data/test_novel.jsonl"):
        forbid.add(normalize(json.loads(l)["in"]))
    for l in open("data/test_real.tsv"):
        if l.strip():
            forbid.add(normalize(l.split("\t")[0]))
    return forbid


def build(n, seed, forbid):
    rng = random.Random(seed)
    rows, rejected = [], 0
    while len(rows) < n:
        t, g = sample(rng)
        if t in forbid or BANNED & set(t.split()) or len(t) > 110:
            rejected += 1
            continue
        rows.append((t, g))
    return rows, rejected


if __name__ == "__main__":
    forbid = load_tests()
    train, rej = build(300000, 11, forbid)
    val, _ = build(2000, 12, forbid)
    print(f"train {len(train)} (rejected {rej} for leakage/banned/length), unique texts {len(set(t for t,_ in train))}")
    for name, rows in (("train_v2", train), ("val_v2", val)):
        with open(f"data/{name}.jsonl", "w") as f:
            for t, g in rows:
                f.write(json.dumps({"in": t, "out": g}) + "\n")
    # convert test_real to jsonl for the shared eval code
    with open("data/test_real.jsonl", "w") as f:
        for l in open("data/test_real.tsv"):
            if l.strip():
                t, g = l.rstrip("\n").split("\t")
                f.write(json.dumps({"in": t, "out": g}) + "\n")
    rng = random.Random(5)
    for t, g in rng.sample(train, 15):
        print(f"  {t!r:62} -> {g}")

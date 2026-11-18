"""
Synthetic dataset: natural English device commands -> compact tool-call strings.

Output grammar (one or more calls joined by '|'):
  light on|off [room=<room>] [delay=<n><s|m|h>]
  fan on|off [speed=<1-3>] [delay=...]
  relay <1-8> on|off [delay=...]
  led <color>|off
  grip [force=<0-100>] | grip adjust=<+20|-20>
  release
  arm home | arm cell=<1-9>
  timer <n><s|m|h> [label=<word>]
  sensor temp|humidity|motion|light
  status [light|fan|arm] [room=<room>]
  cloud                      (anything the tiny model should hand to the cloud LLM)

Phrasings starting with '*' are held out: they never appear in train,
only in the 'test_novel' split, to measure generalization to unseen wording.
"""
import json
import random
import sys

ROOMS = {"kitchen": "kitchen", "bedroom": "bedroom", "living room": "living_room",
         "bathroom": "bathroom", "hall": "hall", "office": "office", "garage": "garage",
         "*hallway": "hall", "*study": "office", "*lounge": "living_room"}
COLORS = ["red", "green", "blue", "white", "yellow", "purple", "orange", "pink", "*cyan", "*magenta"]
NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
             8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 15: "fifteen",
             20: "twenty", 25: "twenty five", 30: "thirty", 40: "forty", 45: "forty five",
             50: "fifty", 60: "sixty", 90: "ninety"}
UNITS = {"s": ["seconds", "secs", "sec", "*s"],
         "m": ["minutes", "mins", "min", "*minute"],
         "h": ["hours", "hrs", "hr", "*hour"]}
LABELS = ["tea", "pasta", "eggs", "laundry", "pizza", "rice", "nap", "break", "*coffee", "*bread"]


def avail(items, split):
    """Pick the phrasing pool for a split. Held-out items start with '*'."""
    normal = [i for i in items if not i.startswith("*")]
    held = [i.lstrip("*") for i in items if i.startswith("*")]
    if split == "train" or not held:
        return normal
    return held + normal if split == "test_novel" else normal


def pick(rng, items, split):
    pool = avail(items, split)
    if split == "test_novel":
        held = [i.lstrip("*") for i in items if i.startswith("*")]
        if held and rng.random() < 0.7:
            return rng.choice(held)
    return rng.choice(pool)


def pick_map(rng, mapping, split):
    key = pick(rng, list(mapping.keys()), split)
    return key, mapping.get(key, mapping.get("*" + key))


def number_text(rng, n):
    if n in NUM_WORDS and rng.random() < 0.4:
        return NUM_WORDS[n]
    return str(n)


def duration(rng, split):
    """Returns (text, canonical) like ('ten minutes', '10m')."""
    special = [("half an hour", "30m"), ("an hour", "1h"), ("a minute", "1m"),
               ("a second", "1s"), ("*a couple of minutes", "2m"), ("*quarter of an hour", "15m")]
    if rng.random() < 0.12:
        txt, can = pick_map(rng, dict(special), split)
        return txt, can
    unit = rng.choice(["s", "m", "m", "h"])
    n = {"s": rng.choice([5, 10, 15, 20, 30, 45, 60, 90]),
         "m": rng.randint(1, 90), "h": rng.randint(1, 12)}[unit]
    return f"{number_text(rng, n)} {pick(rng, UNITS[unit], split)}", f"{n}{unit}"


def delay_phrase(rng, split):
    txt, can = duration(rng, split)
    tpl = pick(rng, ["in {d}", "after {d}", "{d} from now", "*once {d} have passed", "*wait {d} and"], split)
    return tpl.format(d=txt), can, tpl.startswith("wait")


def maybe_room(rng, split, prob=0.4):
    if rng.random() < prob:
        txt, can = pick_map(rng, ROOMS, split)
        return txt, can
    return None, None


# ---------------- intents ----------------

def g_light(rng, split):
    on = rng.random() < 0.5
    if on:
        phr = ["turn on the {L}", "switch on the {L}", "turn the {L} on", "{L} on", "put the {L} on",
               "enable the {L}", "power on the {L}", "i need the {L} on",
               "*flip on the {L}", "*light up the room with the {L}", "*can you get the {L} on"]
    else:
        phr = ["turn off the {L}", "switch off the {L}", "turn the {L} off", "{L} off", "kill the {L}",
               "disable the {L}", "power off the {L}", "shut off the {L}",
               "*cut the {L}", "*can you get the {L} off", "*shut the {L} down"]
    L = pick(rng, ["light", "lights", "lamp", "*lamps", "*lighting"], split)
    room_txt, room = maybe_room(rng, split)
    if room_txt:
        L = rng.choice([f"{room_txt} {L}", f"{L} in the {room_txt}"])
    text = pick(rng, phr, split).format(L=L)
    tgt = f"light {'on' if on else 'off'}" + (f" room={room}" if room else "")
    return text, tgt


def g_fan(rng, split):
    r = rng.random()
    F = pick(rng, ["fan", "ventilator", "*cooler"], split)
    if r < 0.35:
        text = pick(rng, ["turn on the {F}", "start the {F}", "{F} on", "switch the {F} on", "*get the {F} going"], split)
        return text.format(F=F), "fan on"
    if r < 0.7:
        text = pick(rng, ["turn off the {F}", "stop the {F}", "{F} off", "switch the {F} off", "*shut the {F} up"], split)
        return text.format(F=F), "fan off"
    s = rng.randint(1, 3)
    words = {1: ["low", "slow", "minimum"], 2: ["medium", "middle", "*mid"], 3: ["high", "max", "full", "*fast"]}
    if rng.random() < 0.5:
        lvl = pick(rng, words[s], split)
        text = pick(rng, ["set the {F} to {v}", "{F} on {v}", "run the {F} at {v}", "*put the {F} on {v}"], split)
        return text.format(F=F, v=lvl), f"fan speed={s}"
    text = pick(rng, ["set {F} speed to {n}", "{F} speed {n}", "{F} to speed {n}", "*make the {F} speed {n}"], split)
    return text.format(F=F, n=number_text(rng, s)), f"fan speed={s}"


def g_relay(rng, split):
    n = rng.randint(1, 8)
    on = rng.random() < 0.5
    R = pick(rng, ["relay", "switch", "channel", "*output"], split)
    nt = number_text(rng, n)
    if on:
        text = pick(rng, ["turn on {R} {n}", "{R} {n} on", "enable {R} {n}", "close {R} {n}", "*activate {R} {n}"], split)
    else:
        text = pick(rng, ["turn off {R} {n}", "{R} {n} off", "disable {R} {n}", "open {R} {n}", "*deactivate {R} {n}"], split)
    return text.format(R=R, n=nt), f"relay {n} {'on' if on else 'off'}"


def g_led(rng, split):
    if rng.random() < 0.2:
        return pick(rng, ["led off", "turn off the led", "switch the led off", "*kill the led"], split), "led off"
    c = pick(rng, COLORS, split)
    text = pick(rng, ["make the led {c}", "set led to {c}", "led {c}", "change the led color to {c}",
                      "turn the led {c}", "*paint the led {c}", "*i want a {c} led"], split)
    return text.format(c=c), f"led {c}"


def g_grip(rng, split):
    r = rng.random()
    if r < 0.25:
        text = pick(rng, ["grip softer", "loosen the grip", "squeeze less", "less force", "grip a bit softer",
                          "*ease off the grip", "*not so tight"], split)
        return text, "grip adjust=-20"
    if r < 0.5:
        text = pick(rng, ["grip harder", "tighten the grip", "squeeze more", "more force", "grip a bit harder",
                          "*hold it tighter", "*firmer grip"], split)
        return text, "grip adjust=+20"
    if r < 0.75:
        f = rng.choice([10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100])
        text = pick(rng, ["grip with {f} percent force", "grip at {f}%", "set grip force to {f}",
                          "squeeze at {f} percent", "*hold with {f} percent", "*grab it at {f}%"], split)
        return text.format(f=f), f"grip force={f}"
    text = pick(rng, ["grab it", "close the gripper", "grip", "pick it up", "grab the object",
                      "*seize it", "*clamp down"], split)
    return text, "grip"


def g_release(rng, split):
    return pick(rng, ["release", "open the gripper", "let go", "drop it", "release the object", "open the claw",
                      "*put it down", "*let it go"], split), "release"


def g_arm(rng, split):
    if rng.random() < 0.3:
        return pick(rng, ["arm home", "go home", "return the arm home", "reset the arm", "move arm to home",
                          "*park the arm", "*back to start position"], split), "arm home"
    c = rng.randint(1, 9)
    text = pick(rng, ["move to cell {c}", "go to cell {c}", "arm to cell {c}", "move the arm to slot {c}",
                      "reach cell {c}", "*position {c} please", "*head over to cell {c}"], split)
    return text.format(c=number_text(rng, c)), f"arm cell={c}"


def g_timer(rng, split):
    txt, can = duration(rng, split)
    label = None
    if rng.random() < 0.4:
        label = pick(rng, LABELS, split)
        text = pick(rng, ["set a {d} timer for {l}", "{l} timer {d}", "timer for {l} {d}",
                          "remind me about the {l} in {d}", "*start a {d} {l} timer"], split)
        text = text.format(d=txt, l=label)
    else:
        text = pick(rng, ["set a timer for {d}", "timer {d}", "start a {d} timer", "countdown {d}",
                          "*wake me in {d}", "*ping me in {d}"], split).format(d=txt)
    return text, f"timer {can}" + (f" label={label}" if label else "")


def g_sensor(rng, split):
    kinds = {"temp": ["what is the temperature", "how hot is it", "read the temperature", "temperature",
                      "how warm is it", "*is it cold in here", "*check the temp"],
             "humidity": ["what is the humidity", "how humid is it", "read humidity", "humidity level",
                          "*is the air dry", "*check moisture in the air"],
             "motion": ["is there any motion", "did someone move", "check motion sensor", "any movement",
                        "*is anyone there", "*detect movement"],
             "light": ["how bright is it", "read the light sensor", "light level", "what is the brightness",
                       "*is it dark outside", "*measure the light"]}
    k = rng.choice(list(kinds))
    return pick(rng, kinds[k], split), f"sensor {k}"


def g_status(rng, split):
    r = rng.random()
    if r < 0.4:
        room_txt, room = maybe_room(rng, split, 0.5)
        L = "lights" if rng.random() < 0.5 else "light"
        base = pick(rng, ["is the {L} on", "are the {L} on", "check the {L}", "status of the {L}",
                          "*did i leave the {L} on"], split).format(L=L)
        if room_txt:
            base += f" in the {room_txt}"
        return base, "status light" + (f" room={room}" if room else "")
    if r < 0.6:
        return pick(rng, ["is the fan running", "fan status", "is the fan on", "*is the fan spinning"], split), "status fan"
    if r < 0.8:
        return pick(rng, ["where is the arm", "arm status", "arm position", "*what is the arm doing"], split), "status arm"
    return pick(rng, ["status", "system status", "how are you doing", "report status", "*give me a report",
                      "*everything ok"], split), "status"


CLOUD_TPL = ["what is the capital of {x}", "tell me a joke", "who wrote {b}", "how far is the moon",
             "write a poem about {t}", "what is the weather in {x}", "explain {t}", "translate hello to {l}",
             "what is {a} plus {c}", "who is the president of {x}", "recommend a movie", "how do i cook {f}",
             "what time is it in {x}", "summarize the news", "tell me about {t}", "why is the sky blue",
             "book a flight to {x}", "send an email to my boss", "what should i eat today", "play some music",
             "*how does {t} work", "*give me a fun fact", "*what day is it", "*draft a message to my friend",
             "*find a recipe for {f}", "*who invented the {t}"]
FILL = {"x": ["france", "japan", "brazil", "kenya", "canada", "italy", "india", "egypt", "*peru", "*norway"],
        "b": ["hamlet", "dune", "the hobbit", "1984", "*emma"],
        "t": ["black holes", "the ocean", "love", "rockets", "gravity", "*volcanoes", "*the telephone"],
        "l": ["french", "german", "spanish", "*greek"],
        "f": ["rice", "pancakes", "soup", "*lasagna"]}


def g_cloud(rng, split):
    if rng.random() < 0.08:  # random gibberish
        n = rng.randint(3, 20)
        return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz   ") for _ in range(n)).strip() or "hmm", "cloud"
    tpl = pick(rng, CLOUD_TPL, split)
    out = tpl
    for k, vals in FILL.items():
        if "{" + k + "}" in out:
            out = out.replace("{" + k + "}", pick(rng, vals, split))
    out = out.replace("{a}", str(rng.randint(1, 99))).replace("{c}", str(rng.randint(1, 99)))
    return out, "cloud"


SIMPLE = [(g_light, 3), (g_fan, 2), (g_relay, 2), (g_led, 1.5), (g_grip, 2.5), (g_release, 1),
          (g_arm, 2), (g_timer, 2), (g_sensor, 1.5), (g_status, 1.5), (g_cloud, 2.5)]
DELAYABLE = {g_light, g_fan, g_relay}


def one(rng, split):
    fns, ws = zip(*SIMPLE)
    fn = rng.choices(fns, ws)[0]
    text, tgt = fn(rng, split)
    if fn in DELAYABLE and rng.random() < 0.3:
        dtxt, dcan, prefix = delay_phrase(rng, split)
        text = f"{dtxt} {text}" if (prefix or rng.random() < 0.4) else f"{text} {dtxt}"
        tgt += f" delay={dcan}"
    return text, tgt


def sample(rng, split):
    text, tgt = one(rng, split)
    # Chain two commands; cloud never chains.
    if tgt != "cloud" and rng.random() < 0.18:
        t2, g2 = one(rng, split)
        if g2 != "cloud":
            joiner = pick(rng, [" and ", " then ", " and then ", ", ", " also ", "*, after that "], split)
            text, tgt = text + joiner + t2, tgt + "|" + g2
    if rng.random() < 0.15:
        text = pick(rng, ["please ", "can you ", "hey ", "could you ", "*kindly ", "*yo "], split) + text
    if rng.random() < 0.1:
        text += rng.choice([" please", " now", " thanks", "!", "."])
    return " ".join(text.split()), tgt


def build(n, split, seed):
    rng = random.Random(seed)
    return [sample(rng, split) for _ in range(n)]


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data"
    import os
    os.makedirs(out, exist_ok=True)
    sets = {"train": build(80000, "train", 1), "val": build(2000, "val", 2),
            "test_novel": build(2000, "test_novel", 3)}
    train_texts = {t for t, _ in sets["train"]}
    for name, rows in sets.items():
        if name != "train":
            unseen = sum(t not in train_texts for t, _ in rows)
            print(f"{name}: {len(rows)} rows, {unseen} texts never seen in train")
        with open(f"{out}/{name}.jsonl", "w") as f:
            for t, g in rows:
                f.write(json.dumps({"in": t, "out": g}) + "\n")
    print("max input len:", max(len(t) for r in sets.values() for t, _ in r),
          "max output len:", max(len(g) for r in sets.values() for _, g in r))
    for t, g in sets["test_novel"][:12]:
        print(f"  {t!r:60} -> {g}")

import random


def main() -> None:
    random.seed(42)

    animals = [
        ("fox", "fuchs", "m"),
        ("hen", "henne", "f"),
        ("dog", "hund", "m"),
        ("cat", "katze", "f"),
        ("rat", "ratte", "f"),
        ("frog", "frosch", "m"),
        ("horse", "pferd", "n"),
        ("cow", "kuh", "f"),
        ("sheep", "schaf", "n"),
        ("goat", "ziege", "f"),
        ("duck", "ente", "f"),
        ("owl", "eule", "f"),
    ]

    adjectives = [
        ("brown", "braun"),
        ("black", "schwarz"),
        ("white", "weiss"),
        ("gray", "grau"),
        ("red", "rot"),
        ("blue", "blau"),
        ("green", "gruen"),
        ("yellow", "gelb"),
        ("pink", "pink"),
        ("large", "gross"),
        ("small", "klein"),
        ("quick", "schnell"),
    ]

    # Keep one simple translation pair per preposition, and annotate the German case.
    prepositions = [
        ("over", "ueber", "dat"),
        ("under", "unter", "dat"),
        ("around", "um", "acc"),
        ("within", "innerhalb", "gen"),
        ("near", "nahe", "dat"),
        ("behind", "hinter", "dat"),
        ("beside", "neben", "dat"),
        ("across", "entlang", "dat"),
        ("through", "durch", "acc"),
        ("beyond", "jenseits", "gen"),
        ("inside", "in", "dat"),
        ("outside", "ausserhalb", "gen"),
    ]

    locations = [
        ("lake", "see", "m"),
        ("stable", "stall", "m"),
        ("bench", "bank", "f"),
        ("palace", "palast", "m"),
        ("forest", "wald", "m"),
        ("meadow", "wiese", "f"),
        ("river", "fluss", "m"),
        ("bridge", "bruecke", "f"),
        ("garden", "garten", "m"),
        ("barn", "scheune", "f"),
        ("tower", "turm", "m"),
        ("cave", "hoehle", "f"),
    ]

    verbs = [
        ("jumped", "sprang"),
        ("ran", "rannte"),
        ("swam", "schwamm"),
        ("walked", "ging"),
        ("barked", "bellte"),
        ("quacked", "quakte"),
        ("chirped", "zwitscherte"),
        ("slept", "schlief"),
        ("grazed", "graste"),
        ("hunted", "jagte"),
        ("climbed", "kletterte"),
        ("rested", "ruhte"),
    ]

    animal_to_valid_verbs = {
        "fox": {"ran", "jumped", "hunted", "walked", "rested", "slept", "climbed"},
        "hen": {"ran", "walked", "chirped", "rested", "slept"},
        "dog": {"ran", "walked", "barked", "hunted", "rested", "slept", "jumped"},
        "cat": {"ran", "walked", "hunted", "climbed", "rested", "slept", "jumped"},
        "rat": {"ran", "walked", "hunted", "climbed", "rested", "slept"},
        "frog": {"ran", "jumped", "swam", "rested", "slept"},
        "horse": {"ran", "walked", "jumped", "grazed", "rested", "slept"},
        "cow": {"walked", "grazed", "rested", "slept"},
        "sheep": {"walked", "grazed", "rested", "slept", "ran"},
        "goat": {"walked", "grazed", "climbed", "rested", "slept"},
        "duck": {"walked", "swam", "quacked", "rested", "slept"},
        "owl": {"hunted", "rested", "slept", "chirped", "climbed"},
    }

    # Requested extra German determiners.
    de_articles = ["der", "die", "das", "den", "des", "ein", "eine", "einen", "eines", "dem", "einem", "einer"]
    en_articles = ["a", "the"]

    def inflect_adj(base: str, article: str, gender: str, case: str) -> str:
        # Minimal adjective inflection to generate natural-enough training strings.
        if case in {"dat", "gen"}:
            return base + "en"
        if article in {"der", "die", "das"}:
            return base + "e"
        if article in {"ein", "eine"}:
            if gender in {"f"}:
                return base + "e"
            if gender in {"n"}:
                return base + "es"
            return base + "er"
        if article == "einen":
            return base + "en"
        if article in {"dem", "einem", "einer"}:
            return base + "en"
        return base + "en"

    def choose_subject_article(gender: str) -> tuple[str, str]:
        options = [("the", {"m": "der", "f": "die", "n": "das"}[gender])]
        options.append(("a", {"m": "ein", "f": "eine", "n": "ein"}[gender]))
        return random.choice(options)

    def choose_object_article(gender: str, case: str) -> tuple[str, str]:
        if case == "acc":
            def_map = {"m": "den", "f": "die", "n": "das"}
            ind_map = {"m": "einen", "f": "eine", "n": "ein"}
        elif case == "dat":
            def_map = {"m": "dem", "f": "der", "n": "dem"}
            ind_map = {"m": "einem", "f": "einer", "n": "einem"}
        elif case == "gen":
            def_map = {"m": "des", "f": "der", "n": "des"}
            ind_map = {"m": "eines", "f": "einer", "n": "eines"}
        else:
            def_map = {"m": "der", "f": "die", "n": "das"}
            ind_map = {"m": "ein", "f": "eine", "n": "ein"}
        if random.random() < 0.5:
            return "the", def_map[gender]
        return "a", ind_map[gender]

    def inflect_noun(noun: str, gender: str, case: str) -> str:
        # Simple genitive singular for masculine/neuter nouns.
        if case == "gen" and gender in {"m", "n"}:
            return noun + "s"
        return noun

    ordered_vocab = []
    ordered_vocab.extend([x[0] for x in animals])
    ordered_vocab.extend([x[0] for x in adjectives])
    ordered_vocab.extend(en_articles)
    ordered_vocab.extend([x[0] for x in prepositions])
    ordered_vocab.extend([x[0] for x in locations])
    ordered_vocab.extend([x[0] for x in verbs])
    ordered_vocab.extend([x[1] for x in animals])
    ordered_vocab.extend([x[1] for x in prepositions])
    ordered_vocab.extend([x[1] for x in locations])
    ordered_vocab.extend([x[1] for x in verbs])
    ordered_vocab.extend(de_articles)

    # Add genitive noun forms (-s) for masculine/neuter nouns.
    for _, de_noun, gender in animals + locations:
        if gender in {"m", "n"}:
            ordered_vocab.append(de_noun + "s")

    # Add inflected adjective forms (including requested endings en/er/em/es).
    de_adj_bases = [x[1] for x in adjectives]
    for base in de_adj_bases:
        ordered_vocab.append(base)
        for suffix in ("en", "er", "em", "es", "e"):
            ordered_vocab.append(base + suffix)

    # De-duplicate while preserving order.
    ordered_vocab = list(dict.fromkeys(ordered_vocab))

    with open("dummy_vocab.tsv", "w", encoding="utf-8") as f:
        for token_id, word in enumerate(ordered_vocab):
            f.write(f"{word}\t{token_id}\n")

    with open("dummy_train.tsv", "w", encoding="utf-8") as f:
        for _ in range(5000):
            en_an, de_an, an_gender = random.choice(animals)
            en_loc, de_loc, loc_gender = random.choice(locations)
            en_adj_s, de_adj_base_s = random.choice(adjectives)
            en_adj_o, de_adj_base_o = random.choice(adjectives)
            en_prep, de_prep, prep_case = random.choice(prepositions)

            valid_verb_en = random.choice(sorted(animal_to_valid_verbs[en_an]))
            verb_idx = [v[0] for v in verbs].index(valid_verb_en)
            en_v, de_v = verbs[verb_idx]

            en_subj_art, de_subj_art = choose_subject_article(an_gender)
            en_obj_art, de_obj_art = choose_object_article(loc_gender, prep_case)

            de_adj_s = inflect_adj(de_adj_base_s, de_subj_art, an_gender, "nom")
            de_adj_o = inflect_adj(de_adj_base_o, de_obj_art, loc_gender, prep_case)

            de_loc_inflected = inflect_noun(de_loc, loc_gender, prep_case)
            de_sentence = " ".join([de_subj_art, de_adj_s, de_an, de_v, de_prep, de_obj_art, de_adj_o, de_loc_inflected])
            en_sentence = " ".join([en_subj_art, en_adj_s, en_an, en_v, en_prep, en_obj_art, en_adj_o, en_loc])

            f.write(f"{de_sentence}\t{en_sentence}\n")


if __name__ == "__main__":
    main()

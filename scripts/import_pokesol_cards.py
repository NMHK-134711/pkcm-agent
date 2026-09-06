"""Read pokesol parties out of their article markup, ids and all.

    python scripts/import_pokesol_cards.py data/raw/articles/pokesol.app

A pokesol article stores each Pokemon as attributes rather than as a picture
or even as rendered HTML:

    data-pokemon-id data-nature-id data-ability-ids data-item-id
    data-move-ids   data-evs

Moves, items and abilities carry the ids we already share. The species id does
not -- base formes happen to be the national dex number and everything else is
pokesol's own numbering -- so the species is identified from the card instead:
the ability list ends with the *base* ability (a Mega card leads with the
Mega's), the four moves have to be learnable, and a Mega Stone names its
holder outright. Where that leaves one candidate, it is the answer; where it
leaves several the card is held rather than guessed at.

**The cards are not the party.** An article shows the six it used and often
several it only considered, so an importer that assumed six cards meant six
Pokemon would quietly file the rejects. The search index knows which six, and
that is what decides -- the cards supply the sets, the index supplies the team.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pkcm.data.dex import Stat, load_dex  # noqa: E402
from pkcm.engine.legality import (  # noqa: E402
    learnable_moves, mega_stone_for, registrable_abilities, team_errors,
)
from pkcm.engine.pokemon import PokemonSet  # noqa: E402
from pkcm.engine.stats import NATURES, get_nature  # noqa: E402

EV_ORDER = ("hp", "attack", "defense", "specialAttack", "specialDefense", "speed")

#: Formes pokedb spells out and we abbreviate.
SLUG_ALIASES = {
    "tauros-paldea-combat-breed": "taurospaldeacombat",
    "tauros-paldea-blaze-breed": "taurospaldeablaze",
    "tauros-paldea-aqua-breed": "taurospaldeaaqua",
    "basculegion-female": "basculegionf",
    "meowstic-female": "meowsticf",
    "lycanroc-midday": "lycanroc",
}

#: pokesol numbers the natures from 1, grouped by the stat raised (Attack,
#: Defence, Sp. Atk, Sp. Def, Speed) and within each group by the stat lowered
#: in the same order, skipping itself. Derived rather than assumed: over 441
#: cards, id 2 comes with 1132 physical moves and 4 special ones (Adamant), 9
#: with 649 special and 67 physical (Modest), 17 special and 19 physical
#: (Timid and Jolly). Every id lines up with its half of the game.
NATURE_BY_ID = {
    1: "lonely", 2: "adamant", 3: "naughty", 4: "brave",
    5: "bold", 6: "impish", 7: "lax", 8: "relaxed",
    9: "modest", 10: "mild", 11: "rash", 12: "quiet",
    13: "calm", 14: "gentle", 15: "careful", 16: "sassy",
    17: "timid", 18: "hasty", 19: "jolly", 20: "naive",
    21: "serious", 0: "serious",
}


def english(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def tables():
    raw = ROOT / "data" / "raw" / "pokechams"
    species = json.loads((raw / "champions_pokemon.json").read_text(encoding="utf-8"))
    moves = {r["id"]: english(r["nameEn"])
             for r in json.loads((raw / "moves.json").read_text(encoding="utf-8"))}
    items = {r["id"]: english(r["nameEn"])
             for r in json.loads((raw / "items.json").read_text(encoding="utf-8"))}
    abilities = {r["id"]: english(r["nameEn"])
                 for r in json.loads((raw / "abilities.json").read_text(encoding="utf-8"))}
    return species, moves, items, abilities


def cards(page: str) -> list[dict]:
    out = []
    for block in re.findall(r'<div data-type="pokemon-card"([^>]*)></div>', page):
        def attr(name):
            found = re.search(rf'data-{name}="([^"]*)"', block)
            return html.unescape(found.group(1)) if found else None

        evs = attr("evs")
        out.append({
            "pokemon_id": attr("pokemon-id"),
            "nature_id": attr("nature-id"),
            # These arrays carry JavaScript nulls where a slot is empty, which
            # is JSON and is not Python.
            "ability_ids": json.loads(attr("ability-ids") or "[]"),
            "item_id": attr("item-id"),
            "move_ids": json.loads(attr("move-ids") or "[]"),
            "evs": json.loads(evs) if evs else None,
        })
    return out


def identify(card, dex, regulation, items_by_id, abilities_by_id) -> list[str]:
    """Which of our species this card could be. One entry means solved.

    ``items_by_id`` has to be pokesol's own numbering for the Mega Stone
    shortcut to fire. It was pokedb's for a while, which is a different table,
    so no stone was ever recognised and every Mega-holding card fell through to
    being identified by an ability that belongs to the Mega -- 96 articles held
    for want of a Gengar or a Garchomp the card was plainly naming.
    """
    item = items_by_id.get(str(card["item_id"]))

    # A Mega Stone names its holder and nothing else can hold it.
    if item:
        for name in regulation.legal_species:
            if dex.species[name].is_mega:
                continue
            if mega_stone_for(dex, regulation, name) == item:
                return [name]

    # A Mega card leads with the Mega's ability and keeps the base one last.
    base_ability = abilities_by_id.get(str(card["_base_ability"])) \
        if card["_base_ability"] is not None else None
    # A card with one ability carries the Mega's, not the base's, so both
    # ends of the list are tried before giving up on the ability as a filter.
    ends = {abilities_by_id.get(str(one))
            for one in (card["ability_ids"][:1] + card["ability_ids"][-1:])
            if one is not None}
    wanted = {english(mv) for mv in card["_move_names"]}
    found = []
    for name in sorted(regulation.legal_species):
        if dex.species[name].is_mega:
            continue
        allowed = set(registrable_abilities(dex.species[name]) or ())
        if ends and allowed and not (ends & allowed):
            continue
        if not wanted <= learnable_moves(dex, name):
            continue
        found.append(name)
    return found


def main() -> int:
    for out in (sys.stdout, sys.stderr):
        try:
            out.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--index", default=str(ROOT / "data" / "champions" / "articles.json"))
    parser.add_argument("--regulation", default="m_b")
    parser.add_argument("--out", default=str(ROOT / "runs" / "pokesol_parties.json"))
    args = parser.parse_args()

    dex = load_dex()
    regulation = dex.regulation(args.regulation)
    species_rows, moves_by_id, items_by_id, abilities_by_id = tables()
    # pokesol's own item numbering, learned by an earlier run of this script.
    # Absent on the first run, which is why it takes two to settle.
    # pokesol numbers moves its own way too. pokedb's Archaludon page keys
    # Electro Shot at 905 and so does our table; pokesol's cards say 907. The
    # low ids agreeing was taken for the whole range agreeing, which it was
    # not, and six ids had to be read off the articles to settle.
    corrections = ROOT / "data" / "champions" / "pokesol_moves.json"
    if corrections.exists():
        fixes = {k: v["id"] for k, v in
                 json.loads(corrections.read_text(encoding="utf-8")).items()
                 if not k.startswith("_")}
        moves_by_id = {**moves_by_id, **fixes}
        print(f"applying {len(fixes)} corrected pokesol move ids")

    # Sets a person read off an article the cards could not settle. They are
    # filled in before completeness is judged and still go through
    # team_errors: a hand-entered set is not exempt from the check, and the
    # one that started this project was a Primarina holding a move it did not
    # have.
    by_hand: dict[str, dict] = {}
    hand = ROOT / "data" / "champions" / "manual_sets.json"
    if hand.exists():
        by_hand = {k: v for k, v in
                   json.loads(hand.read_text(encoding="utf-8")).items()
                   if not k.startswith("_")}
        print(f"{sum(len(v) for v in by_hand.values())} hand-read sets "
              f"across {len(by_hand)} articles")

    learned_items = ROOT / "data" / "champions" / "pokesol_items.json"
    if learned_items.exists():
        table = json.loads(learned_items.read_text(encoding="utf-8"))
        items_by_id = {ident: row["id"] for ident, row in table.items()}
        print(f"using {len(items_by_id)} learned pokesol item ids")

    index = {a["url"]: a for a in
             json.loads(Path(args.index).read_text(encoding="utf-8"))["articles"]}
    by_slug = {}
    for url in index:
        tail = re.sub(r"[^A-Za-z0-9._-]", "_",
                      url.split("://", 1)[1].split("/", 1)[1].strip("/"))
        by_slug[tail[:120] + ".html"] = url

    files: list[Path] = []
    for one in args.paths:
        path = Path(one)
        files.extend(sorted(path.glob("*.html")) if path.is_dir() else [path])

    # Two passes. The first identifies what the card's own contents settle
    # and learns what each pokesol species id means from those; the second
    # spends that on the cards the contents left ambiguous -- two formes with
    # the same ability and the same learnset, mostly.
    learned: dict[str, set] = {}
    for path in files:
        page = path.read_text(encoding="utf-8", errors="replace")
        for card in cards(page):
            names = [moves_by_id.get(str(one), "") if one is not None else ""
                     for one in card["move_ids"]]
            tail = [one for one in card["ability_ids"] if one is not None]
            card["_move_names"] = names
            card["_base_ability"] = tail[-1] if tail else None
            if len(names) != 4 or not all(names) or not card["evs"]:
                continue
            options = identify(card, dex, regulation, items_by_id, abilities_by_id)
            if len(options) == 1 and card["pokemon_id"]:
                learned.setdefault(card["pokemon_id"], set()).add(options[0])
    by_pokesol_id = {one: next(iter(names)) for one, names in learned.items()
                     if len(names) == 1}
    print(f"pokesol species ids learned: {len(by_pokesol_id)} "
          f"(of {len(learned)} seen)")

    # pokesol numbers items its own way -- not the dex's, and not pokedb's
    # either. The index names each article's six in the order of its six
    # species, so once the species are matched the numbers are too.
    item_votes: dict[str, Counter] = {}
    kept, held = [], []
    stats = {"cards": 0, "identified": 0, "ambiguous": 0, "by_elimination": 0}
    for path in files:
        page = path.read_text(encoding="utf-8", errors="replace")
        url = by_slug.get(path.name)
        row = index.get(url or "")
        found, found_item, problems, spare = {}, {}, [], []
        for card in cards(page):
            stats["cards"] += 1
            names = [moves_by_id.get(str(one), "") if one is not None else ""
                     for one in card["move_ids"]]
            tail = [one for one in card["ability_ids"] if one is not None]
            card["_move_names"] = names
            card["_base_ability"] = tail[-1] if tail else None
            if len(names) != 4 or not all(names) or not card["evs"]:
                continue
            options = identify(card, dex, regulation, items_by_id, abilities_by_id)
            if len(options) != 1:
                known = by_pokesol_id.get(card["pokemon_id"])
                if known is None or (options and known not in options):
                    stats["ambiguous"] += 1
                    spare.append(card)
                    continue
                options = [known]
            stats["identified"] += 1
            name = options[0]
            nature = NATURE_BY_ID.get(int(card["nature_id"] or 0), "serious")
            # The card names the ability the Pokemon fights with, which for
            # a Mega is the Mega's. The set registers the base, and the base
            # cannot hold Shadow Tag -- so an ability it may not take falls
            # back to one it may.
            ability = abilities_by_id.get(str(card["_base_ability"]), "__none__")
            allowed = registrable_abilities(dex.species[name]) or ()
            if allowed and ability not in allowed:
                ability = allowed[0]
            found[name] = PokemonSet(
                species=name,
                ability=ability,
                moves=tuple(card["_move_names"]),
                item=items_by_id.get(str(card["item_id"])),
                nature=nature,
                sp=tuple(int(card["evs"].get(key, 0)) for key in EV_ORDER))
            found_item[name] = card["item_id"]

        if row is None:
            problems.append("no index row for this article")
            team = list(found.values())[:6]
        else:
            want = []
            for entry in row["team"]:
                key = entry["dex"]
                match = next((r for r in species_rows
                              if f"{int(r['nationalDex']):04d}-{int(r['formIndex']):02d}" == key),
                             None)
                if match is None:
                    # Some formes have no row at all -- Mega Pyroar is 0668-02
                    # and the species table simply lacks it. The dex number
                    # still names the family, and a Mega registers as its base.
                    number = int(key.split("-")[0])
                    kin = [one for one in regulation.legal_species
                           if not dex.species[one].is_mega
                           and dex.species[one].dex_num == number]
                    if len(kin) == 1:
                        want.append(kin[0])
                    continue
                # pokedb spells a Mega by repeating the name --
                # metagross-mega-metagross -- while a regional forme is the
                # plain compound our ids already use. A Mega registers as its
                # base holding the stone either way, so the prefix is what we
                # want. Getting this wrong silently held 183 of 241 parties
                # looking for a species that does not exist.
                # pokedb spells a Mega either as floette-mega or as
                # metagross-mega-metagross, so both shapes have to fold to the
                # base -- and the base is what registers, holding the stone.
                slug = re.sub(r"-mega(-.*)?$", "", match["slug"])
                name = slug.replace("-", "")
                if name not in dex.species:
                    name = slug
                if name not in dex.species:
                    # pokedb spells out what our ids abbreviate.
                    name = SLUG_ALIASES.get(match["slug"], name)
                if name not in dex.species:
                    continue
                base = dex.species[name]
                if base.is_mega and base.base_species:
                    name = base.base_species
                # Folding a forme can land on one the regulation does not
                # allow: Mega Floette is legal, plain Floette is not, and only
                # Floette-Eternal is registrable. Take the legal sibling.
                if name not in regulation.legal_species:
                    kin = [one for one in regulation.legal_species
                           if not dex.species[one].is_mega
                           and dex.species[one].dex_num == base.dex_num]
                    if len(kin) == 1:
                        name = kin[0]
                want.append(name)
            for name, entry in (by_hand.get(url or "") or {}).items():
                # A leading underscore is a note, not a set -- "_unavailable"
                # is how an article says the builder never published one.
                if name.startswith("_") or name in found:
                    continue
                # A reading beats the index. pokedb has the rank 79 Tauros as
                # the Combat breed and the article says Aqua, and Raging Bull
                # changes type with the breed, so this is not cosmetic. Where
                # the hand-read species is a sibling of one the index named,
                # the sibling steps aside.
                if name not in want:
                    number = dex.species[name].dex_num
                    for slot, other in enumerate(want):
                        if dex.species[other].dex_num == number:
                            want[slot] = name
                            found.pop(other, None)
                            break
                found[name] = PokemonSet(
                    species=name, ability=entry["ability"],
                    moves=tuple(entry["moves"]), item=entry["item"],
                    nature=entry["nature"], sp=tuple(entry["sp"]))
                found_item[name] = None
            missing = [name for name in want if name not in found]
            # One species left over and one card left over is not a guess: the
            # index says the party is these six, the cards account for five,
            # and the last card is the last species. The learnset still has to
            # agree, so a card that could not be that Pokemon is left alone.
            if len(missing) == 1 and len(spare) == 1:
                card, name = spare[0], missing[0]
                if {english(mv) for mv in card["_move_names"]} <= learnable_moves(dex, name):
                    found[name] = PokemonSet(
                        species=name,
                        ability=abilities_by_id.get(str(card["_base_ability"]), "__none__"),
                        moves=tuple(card["_move_names"]),
                        item=items_by_id.get(str(card["item_id"])),
                        nature=NATURE_BY_ID.get(int(card["nature_id"] or 0), "serious"),
                        sp=tuple(int(card["evs"].get(key, 0)) for key in EV_ORDER))
                    found_item[name] = card["item_id"]
                    stats["by_elimination"] += 1
                    missing = []
            team = [found[name] for name in want if name in found]
            if missing:
                problems.append(f"no card for {', '.join(missing)}")

        if len(team) != 6:
            problems.append(f"built {len(team)} of 6")
        else:
            problems.extend(team_errors(dex, regulation, tuple(team)))
        if row is not None and len(team) == 6:
            for one, entry in zip(team, row["items"]):
                ident = found_item.get(one.species)
                if ident is not None and entry.get("name"):
                    item_votes.setdefault(str(ident), Counter())[entry["name"]] += 1

        record = {"source": path.name, "url": url,
                  "rank": (row or {}).get("rank"), "rating": (row or {}).get("rating"),
                  "team": [{"species": one.species, "ability": one.ability,
                            "moves": list(one.moves), "item": one.item,
                            "nature": one.nature, "sp": list(one.sp),
                            "_item_id": str(found_item.get(one.species, ""))}
                           for one in team],
                  "problems": problems}
        (kept if not problems else held).append(record)

    # Japanese name -> our id, so the learned numbering lands on real items.
    ja_to_id = {r["nameJa"]: english(r["nameEn"])
                for r in json.loads(
                    (ROOT / "data" / "raw" / "pokechams" / "items.json")
                    .read_text(encoding="utf-8"))}
    item_table, unnamed = {}, 0
    for ident, votes in item_votes.items():
        name_ja, count = votes.most_common(1)[0]
        ours = ja_to_id.get(name_ja)
        if ours is None:
            unnamed += 1
            continue
        item_table[ident] = {"id": ours, "name_ja": name_ja, "seen": count,
                             "agreement": round(count / sum(votes.values()), 3)}
    (ROOT / "data" / "champions" / "pokesol_items.json").write_text(
        json.dumps(item_table, ensure_ascii=False, indent=1, sort_keys=True),
        encoding="utf-8")
    print(f"pokesol item ids learned: {len(item_table)} "
          f"({unnamed} named something we do not carry)")
    shaky = [k for k, v in item_table.items() if v["agreement"] < 0.9]
    if shaky:
        print(f"   {len(shaky)} with the six not agreeing every time")

    for record in kept + held:
        for one in record["team"]:
            if one["item"] is None:
                learnt = item_table.get(str(one.pop("_item_id", "")), None)
                if learnt:
                    one["item"] = learnt["id"]

    Path(args.out).write_text(
        json.dumps({"parties": kept, "held": held}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"articles {len(files)}   cards {stats['cards']}   "
          f"identified {stats['identified']}   ambiguous {stats['ambiguous']}   "
          f"by elimination {stats['by_elimination']}")
    print(f"clean parties {len(kept)}   held {len(held)}   -> {args.out}")
    reasons = {}
    for record in held:
        reasons[record["problems"][0].split(":")[0][:44]] = \
            reasons.get(record["problems"][0].split(":")[0][:44], 0) + 1
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1])[:8]:
        print(f"   {count:4}  {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

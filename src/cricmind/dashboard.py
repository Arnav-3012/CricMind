"""Local Streamlit viewer for the Phase 6 tools (D-063): the same Toolbox the agent calls, one tab per tool.

uv run --env-file .env streamlit run src/cricmind/dashboard.py

No modelling of its own: every probability and count is a tool result, so what you see is what the agent can
quote. Every player choice is a searchable dropdown (type to filter) keyed by registry id. The Clusters tab
refits nothing: it re-reads the saved role clusters and projects the train vectors to 2-D for display.
The "Ask the agent" tab needs LLM_* in .env; the other tabs work without a key. Not the Phase 7 UI (D-019).
"""

import json

import altair as alt
import pandas as pd
import streamlit as st

from cricmind import config
from cricmind.agent.agent import run
from cricmind.agent.llm import LLM
from cricmind.agent.prompts import REFUSAL
from cricmind.agent.serving import Serving
from cricmind.agent.tools import Toolbox
from cricmind.data.db import connect
from cricmind.eval.agent_eval import answer_ok
from cricmind.features.context import PHASES

st.set_page_config(page_title="CricMind", layout="wide")
QUESTION_FILES = ["agent_questions.jsonl", "agent_questions_tools.jsonl"]
OUTCOMES = config.OUTCOME_CLASSES


@st.cache_resource
def toolbox() -> Toolbox:
    return Toolbox(connect(), Serving())


@st.cache_data
def player_options(role: str) -> dict[str, str]:
    """id -> label. The id is the key (names collide); the ball count tells namesakes apart."""
    return {r["id"]: f"{r['name']} ({r['balls']:,} balls, {r['id']})" for r in toolbox().players(role)}


@st.cache_data
def pair_options() -> list[dict]:
    return toolbox().pairs()


@st.cache_data
def questions() -> list[dict]:
    rows = []
    for name in QUESTION_FILES:
        rows += [json.loads(line) for line in (config.ROOT / "evals" / name).read_text().splitlines()]
    return rows


def player_dropdown(label: str, role: str, key: str, index: int = 0) -> str:
    options = player_options(role)
    return st.selectbox(label, list(options), index=index, format_func=options.get, key=key)


PHASE_WORDS = {
    "powerplay": "powerplay (overs 1-6)",
    "middle": "middle overs (7-15)",
    "death": "death overs (16-20)",
}


def phase_shares(profile: dict | None) -> pd.DataFrame | None:
    """Share of a player's career balls in each phase, from the player_profile tool (all seasons)."""
    if not profile:
        return None
    rows = [{"phase": p, "balls": v["balls"]} for p, v in profile["by_phase"].items()]
    df = pd.DataFrame(rows)
    df["share"] = df["balls"] / df["balls"].sum()
    return df


def trait_sentence(role: str, feature: str, z: float, mean: float) -> str:
    """One plain sentence for one bin trait: `mean` is the bin's average, the sign of z the direction."""
    more = "more" if z > 0 else "less"
    count = "More" if z > 0 else "Fewer"
    if feature.startswith("share_"):
        phase = PHASE_WORDS[feature.removeprefix("share_")]
        return f"Spends {more} of their balls in the {phase} than most: about {mean:.0%} of their balls."
    if feature == "entry_over":
        return f"Comes in {'later' if z > 0 else 'earlier'} than most: around over {mean:.1f} on average."
    if feature == "entry_wickets":
        return (
            f"Walks in {'with more' if z > 0 else 'with fewer'} wickets down than most: "
            f"about {mean:.1f} wickets on average."
        )
    kind, _, phase = feature.rpartition("_")
    where = PHASE_WORDS[phase]
    if kind == "strike_rate":
        what = "scores" if role == "batter" else "concedes"
        unit = "runs per 100 balls"
        return f"{what.title()} {more} than usual in the {where}: about {mean:.0f} {unit}."
    if kind == "dot_pct":
        return f"{count} dot balls than usual in the {where}: about {mean:.0f}% of balls."
    if kind == "boundary_pct":
        who = "hits" if role == "batter" else "concedes"
        return f"{count} boundaries than usual in the {where}: {who} one on about {mean:.0f}% of balls."
    return f"{feature.replace('_', ' ')}: {more} than average (z {z:+.2f})."


def explain_archetype(role: str, traits: dict, profile: dict | None) -> list[str]:
    """Bullets: the bin's distinctive traits in words, then where this player's own balls fall."""
    lines = [trait_sentence(role, f, v["z"], v["mean"]) for f, v in traits.items()]
    shares = phase_shares(profile)
    if shares is not None:
        top = shares.sort_values("share", ascending=False).iloc[0]
        lines.append(
            f"This player's own career: mostly the {PHASE_WORDS[top['phase']]}, "
            f"{top['share']:.0%} of {int(shares['balls'].sum()):,} balls."
        )
    return lines


def bin_name(role: str, label: int) -> str:
    """A description built from the bin's three most distinctive traits (D-056(4): bins are numbered, so this
    is a reading of the numbers, not a validated cricket role)."""
    traits = toolbox().srv.manifest["roles"][role]["clusters"][str(int(label))]["top"]
    words = [f"{'high' if v['z'] > 0 else 'low'} {f.replace('_', ' ')}" for f, v in list(traits.items())[:2]]
    return f"{role.title()} bin {int(label)}: " + ", ".join(words)


@st.cache_data(show_spinner="Projecting the train vectors (first load takes a moment)...")
def role_map(role: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(2-D points per clustered player, bin x feature z-score table). Train rows only, as the clustering."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    from cricmind.archetypes import vectors as vec
    from cricmind.archetypes.roles import role_vectors

    srv = toolbox().srv
    lab = srv.labels[(srv.labels["role"] == role) & (srv.labels["placed"] == "clustered")]
    train, _ = vec.load_train()
    X = role_vectors(train, role, pd.Index(lab["player_id"]))
    Z = StandardScaler().fit_transform(X)
    xy = PCA(2, random_state=config.SEED).fit_transform(Z)
    names = toolbox().con.execute("select player_id, name from players").df().set_index("player_id")["name"]
    labels = lab.reset_index(drop=True)  # role_vectors keeps the order of the ids it is given
    points = pd.DataFrame(
        {
            "PC1": xy[:, 0],
            "PC2": xy[:, 1],
            "player": names.reindex(labels["player_id"]).to_numpy(),
            "train_balls": labels["train_balls"].to_numpy(),
            "archetype": [bin_name(role, v) for v in labels["label"]],
        }
    )
    z = pd.DataFrame(Z, columns=X.columns).groupby(labels["label"].to_numpy()).mean()
    z.index = [bin_name(role, v) for v in z.index]
    return points, z


def probability_chart(table: pd.DataFrame) -> alt.Chart:
    """Outcome probabilities, one group of bars per outcome class, in the class order of config."""
    long = table.rename_axis("outcome").reset_index().melt("outcome", var_name="series", value_name="p")
    return (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X("outcome:N", sort=OUTCOMES), y=alt.Y("p:Q", title="probability"),
            color="series:N", xOffset="series:N",
            tooltip=["outcome", "series", alt.Tooltip("p:Q", format=".4f")],
        )
    )  # fmt: skip


box = toolbox()
st.title("CricMind: batter vs bowler")
st.caption(
    "Model numbers come from models fitted on seasons up to 2023 and scored on later seasons. "
    "Observed counts cover all seasons. Cricsheet data, ODC-By 1.0."
)
tab_match, tab_clusters, tab_arch, tab_next, tab_agent = st.tabs(
    ["Matchup", "Clusters", "Archetype", "Next ball", "Ask the agent"]
)

with tab_match:
    mode = st.radio("Choose", ["Most-faced pairs", "Any batter and bowler"], horizontal=True, key="m_mode")
    if mode == "Most-faced pairs":
        pairs = {(r["batter_id"], r["bowler_id"]): r for r in pair_options()}
        chosen = st.selectbox(
            "Pair",
            list(pairs),
            format_func=lambda k: f"{pairs[k]['batter']} vs {pairs[k]['bowler']} ({pairs[k]['balls']} balls)",
            key="m_pair",
        )
        batter, bowler = chosen
    else:
        left, right = st.columns(2)
        with left:
            batter = player_dropdown("Batter", "batter", "m_bat")
        with right:
            bowler = player_dropdown("Bowler", "bowler", "m_bowl", index=1)
    phase = st.radio("Phase", ["all", *PHASES], horizontal=True, key="m_phase")
    res = box.matchup(batter, bowler, None if phase == "all" else phase)
    st.subheader(f"{res['batter']['name']} vs {res['bowler']['name']}")
    obs, mod = st.columns(2)
    with obs:
        st.markdown("**Observed (all seasons, legal balls)**")
        o = {k: v for k, v in res["observed"].items() if k != "by_phase"}
        st.dataframe(pd.Series(o, name="value"))
    with mod:
        st.markdown("**Model probability of the next ball** (train seasons, <= 2023)")
        table = pd.DataFrame({ph: pd.Series(v["p"]) for ph, v in res["model"].items()})
        st.altair_chart(probability_chart(table), use_container_width=True)
    st.markdown("**Train balls behind each level** (0 = that level was skipped, the parent passes through)")
    st.dataframe(pd.DataFrame({ph: v["train_balls"] for ph, v in res["model"].items()}))
    st.caption(res["model_note"])

with tab_clusters:
    role = st.radio("Role", ["batter", "bowler"], horizontal=True, key="c_role")
    info = box.srv.manifest["roles"][role]
    points, z = role_map(role)
    st.markdown(
        f"**{role.title()} role bins: k = {info['k']}, bootstrap ARI {info['boot_ari']:.3f}, "
        f"{'stable' if info['stable'] else 'NOT stable'}**; {len(points):,} clustered players."
    )
    if not info["stable"]:
        st.warning("These bins change when the data is resampled: read them as a description.")
    st.altair_chart(
        alt.Chart(points)
        .mark_circle(opacity=0.7)
        .encode(
            x="PC1:Q", y="PC2:Q", color=alt.Color("archetype:N", title="Archetype"),
            size=alt.Size("train_balls:Q", legend=None, scale=alt.Scale(range=[20, 200])),
            tooltip=["player", "archetype", "train_balls"],
        )
        .interactive(),
        use_container_width=True,
    )  # fmt: skip
    st.caption("2-D PCA of the standardised role vectors, for display; the clustering used all dimensions.")
    heat = z.rename_axis("archetype").reset_index().melt("archetype", var_name="feature", value_name="z")
    st.altair_chart(
        alt.Chart(heat)
        .mark_rect()
        .encode(
            x=alt.X("feature:N", title=None), y=alt.Y("archetype:N", title=None),
            color=alt.Color("z:Q", scale=alt.Scale(scheme="redblue", domain=[-1.5, 1.5], reverse=True)),
            tooltip=["archetype", "feature", alt.Tooltip("z:Q", format=".2f")],
        ),
        use_container_width=True,
    )  # fmt: skip
    st.caption("Mean z-score of each feature per bin: red = above the population, blue = below.")
    sizes = points["archetype"].value_counts().rename("players")
    st.dataframe(sizes)
    st.caption(
        "Archetype names are generated from each bin's two most distinctive traits (D-056(4): bins are "
        "numbered, not validated roles). Players with 50+ train balls but not clustered are placed on the "
        "nearest centre and are not drawn."
    )

with tab_arch:
    role = st.radio("Show the player as a", ["batter", "bowler"], horizontal=True, key="a_role")
    player = player_dropdown(f"{role.title()}", role, f"a_player_{role}")  # only players who have that role
    res = box.archetype_of(player)
    got = res.get(role)
    st.subheader(f"{res['name']}, as a {role}")
    if got is None:
        floor = config.MIN_TRAIN_BALLS_EMBEDDING
        st.info(f"No {role} archetype: fewer than {floor} train balls in this role.")
    else:
        a, b, c = st.columns(3)
        a.metric("Archetype", f"Bin {got['bin']}")
        b.metric("Train balls", f"{got['train_balls']:,}")
        c.metric("Players in the bin", f"{got['clustered_players_in_bin']:,}")
        st.markdown(f"**{bin_name(role, got['bin'])}**")
        how = "clustered directly" if got["placed"] == "clustered" else "placed on the nearest cluster centre"
        st.caption(f"This player was {how}. Bins are numbered; the name is built from the bin's traits.")

        st.markdown("#### In plain English")
        profile = box.player_profile(player)[{"batter": "batting", "bowler": "bowling"}[role]]
        for line in explain_archetype(role, got["bin_traits"], profile):
            st.markdown(f"- {line}")

        shares = phase_shares(profile)
        if shares is not None:
            st.markdown("#### Where they play (career, all seasons)")
            st.altair_chart(
                alt.Chart(shares)
                .mark_bar()
                .encode(
                    x=alt.X("phase:N", sort=list(PHASES), title=None),
                    y=alt.Y("share:Q", title="share of balls", axis=alt.Axis(format="%")),
                    tooltip=["phase", alt.Tooltip("share:Q", format=".0%"), "balls"],
                ),
                use_container_width=True,
            )

        with st.expander("The numbers behind the bin"):
            st.caption("Each trait: the bin's average value and how far it sits from the average player (z).")
            traits = pd.DataFrame(got["bin_traits"]).T.rename(columns={"z": "z vs average player"})
            st.dataframe(traits)
        if not got["clustering_stable"]:
            st.warning(
                f"{role.title()} bins are not stable (bootstrap ARI {got['bootstrap_ari']}): "
                "treat this as a description of how they played, not a fixed type."
            )
    st.caption(res["note"])

with tab_next:
    left, right = st.columns(2)
    with left:
        batter = player_dropdown("Batter", "batter", "n_bat")
    with right:
        bowler = player_dropdown("Bowler", "bowler", "n_bowl", index=1)
    c1, c2, c3 = st.columns(3)
    over = c1.slider("Over (0-19)", 0, config.SCHEDULED_OVERS - 1, 0)
    innings = c2.radio("Innings", [1, 2], horizontal=True)
    comps = ["(default)", *box.srv.manifest["next_ball"]["levels"]["competition"]]
    comp = c3.selectbox("Competition", comps)
    state = {}
    if st.checkbox("Set match state (otherwise train medians are used)"):
        s1, s2, s3 = st.columns(3)
        state = {
            "score_before": s1.number_input("Score", 0, 300, 0),
            "wickets_before": s2.number_input("Wickets", 0, 9, 0),
            "batter_runs": s3.number_input("Batter runs", 0, 200, 0),
            "batter_balls_faced": s1.number_input("Batter balls faced", 0, 120, 0),
            "bowler_balls_bowled": s2.number_input("Bowler balls bowled", 0, 24, 0),
        }
        if innings == 2:
            state["target"] = s3.number_input("Target", 1, 300, 160)
    try:
        league = None if comp == "(default)" else comp
        res = box.predict_next_ball(batter, bowler, over, innings, league, **state)
        chart = probability_chart(pd.DataFrame({"model": pd.Series(res["p"])}))
        st.altair_chart(chart, use_container_width=True)
        st.write("Inputs filled in by the tool:", res["assumed"] or "none")
        st.caption(res["model_note"])
    except ValueError as e:
        st.error(str(e))

with tab_agent:
    bank = {f"{q['id']} [{q['category']}] {q['question']}": q for q in questions()}
    choice = st.selectbox("Question", ["(type my own)", *bank], key="q_choice")
    q = bank.get(choice)
    question = q["question"] if q else st.text_input("Your question")
    if q and q["refuse"]:
        st.caption("Expected: a refusal (the data cannot answer this).")
    if question and st.button("Ask"):
        try:
            llm = LLM()
        except RuntimeError as e:
            st.error(str(e))
        else:
            with st.spinner("Calling tools..."):
                out = run(question, llm, box)
            st.write(out["answer"] or "(no answer: the step or token cap was reached)")
            st.caption(f"{len(out['calls'])} tool calls, {out['tokens']:,} tokens, {out['latency']} s")
            if q:  # a bank question has a truth: show the same check the eval applies
                if q["refuse"]:
                    refused = out["answer"].strip().startswith(REFUSAL)
                    st.write("Refused as expected." if refused else "Did not refuse.")
                else:
                    verdict = answer_ok(q, out["answer"])
                    shown = {None: "no truth to check", True: "match", False: "no match"}[verdict]
                    st.write(f"Expected {q['truth']} (tolerance {q['tolerance']}): {shown}")
            st.json(out["calls"], expanded=False)

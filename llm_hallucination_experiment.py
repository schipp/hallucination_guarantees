import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from statsmodels.stats.proportion import proportion_confint
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# plotting params
plt.rcParams["font.family"] = "serif"
plt.style.use(Path("colorblind_friendly.mplstyle"))

# load LLM
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype="auto", device_map="auto"
)
model = torch.compile(model)

# Example tasks as in Appendix A
TASKS = [
    {
        "A": "Sentence one: She wore a bright red striped shirt to the party.\nSentence two: It was very stylish.",
        "c0": "striped",
        "target": "sixth",
    },
    {
        "A": "Sentence one: Information about the new project was sent to all staff members.\nSentence two: Please check your email.",
        "c0": "sent",
        "target": "seventh",
    },
    {
        "A": "Sentence one: Those three small round silver coins were found in the dirt.\nSentence two: They looked like ancient Roman money.",
        "c0": "found",
        "target": "eighth",
    },
]


def task_prompt(A, target):
    """Task prompt template as in Appendix A."""
    return f"""Below are two sentences.

    Your task: From the *first* sentence, find the {target} word and respond with only that word.

    Note: The first sentence has many words, including some that are distractors. The key word is near the middle but might be obscured by distractors or complex phrasing. If the answer is clear, give the {target} word. If you are unsure or find it too confusing, say "I don't know".

    Here are the sentences:

    {A}

    Respond only with the {target} word of the first sentence, or "I don't know" if you're unsure.
    """


def issue_batch_to_qwen(N, prompt):
    """Issue a batch of N LLM calls in parallel while ensuring independence."""
    messages = [[{"role": "user", "content": prompt}] for _ in range(N)]
    texts = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    outputs = model.generate(**model_inputs, max_new_tokens=16)

    answers = []
    for i in range(N):
        gen_ids = outputs[i][len(model_inputs.input_ids[i]) :]
        answers.append(
            tokenizer.decode(gen_ids, skip_special_tokens=True).strip().lower()
        )
    return answers


def compute_wilson_ci(frates, R, alpha=0.05):
    """Compute Wilson confidence intervals for given failure rates."""
    lower = torch.zeros_like(frates)
    upper = torch.zeros_like(frates)
    for i, frate in enumerate(frates):
        k = int(frate * R)
        lower[i], upper[i] = proportion_confint(
            count=k, nobs=R, alpha=alpha, method="wilson"
        )
    return lower, upper


def Q_majority_torch(q, k):
    """Compute Q for given q and odd k. Used for Theorem 3.1."""
    q = torch.tensor(q, dtype=torch.float32)
    m = (k // 2) + 1
    i_vals = torch.arange(m, k + 1, dtype=torch.float32)
    combs = torch.tensor([math.comb(k, int(i)) for i in i_vals], dtype=torch.float32)
    q_term = torch.pow(q.unsqueeze(-1), i_vals)
    one_minus_q = torch.pow(1 - q.unsqueeze(-1), k - i_vals)
    probs = combs * q_term * one_minus_q
    return probs.sum(dim=-1)


def judge_only_hallucination_rate_torch(q_pp, q_np, k):
    """Compute hallucination-selection rate for given q++/q-+ and odd k. Used for Theorem 3.1."""
    Q_pp = Q_majority_torch(q_pp, k)
    Q_np = Q_majority_torch(q_np, k)
    return Q_np / (Q_pp + Q_np)


def judge_only_hallucination_rate_all_ks(q_pp, q_np, ks):
    """Compute hallucination-selection rates for all ks. Used for Theorem 3.1."""
    return torch.tensor(
        [judge_only_hallucination_rate_torch(q_pp, q_np, k).item() for k in ks]
    )


def run_experiment(task, N_max=6, R=200, k_max=9, flip_prob=0.20):
    """Run the full LLM pipeline experiment for a given task."""
    odd_ks = list(range(1, k_max + 1, 2))
    num_ks = len(odd_ks)

    # Generate the task prompt
    prompt = task_prompt(task["A"], task["target"])
    c0 = task["c0"].lower()

    # Initialize statistics
    empirical_p_count = 0.0
    pipeline_failure = torch.zeros(N_max)
    hallucination_selection = torch.zeros((N_max, num_ks))
    pipeline_success = torch.zeros((N_max, num_ks))
    tp_s = fp_s = tn_s = fn_s = 0
    TP_k = torch.zeros(num_ks)
    FN_k = torch.zeros(num_ks)
    FP_k = torch.zeros(num_ks)
    TN_k = torch.zeros(num_ks)

    # Run R repetitions of the experiment to gather statistics
    for _ in tqdm(range(R), desc="Running Pipeline"):
        answers = issue_batch_to_qwen(N_max, prompt)
        # Determine correctness of each answer
        is_correct = torch.tensor([a == c0 for a in answers], dtype=torch.bool)

        # Update empirical p count
        empirical_p_count += (~is_correct).sum().item()

        # Simulate synthetic base judge with error rate flip_prob
        single_votes = is_correct ^ (torch.rand(N_max) < flip_prob)

        # Update base judge stats
        tp_s += torch.sum(single_votes & is_correct).item()
        fp_s += torch.sum(single_votes & ~is_correct).item()
        tn_s += torch.sum(~single_votes & ~is_correct).item()
        fn_s += torch.sum(~single_votes & is_correct).item()

        for n_idx in range(N_max):
            sv = single_votes[: n_idx + 1]
            correct_n = is_correct[: n_idx + 1]

            # Check for pipeline failure (no judged-correct answers)
            if not sv.any():
                pipeline_failure[n_idx] += 1
                continue

            # Set of judged-correct answer indices
            S_star_indices = torch.nonzero(sv, as_tuple=False).flatten()

            for k_idx, k in enumerate(odd_ks):
                # generate k noisy votes for each answer in S*
                votes = correct_n[S_star_indices].unsqueeze(1).expand(-1, k)
                votes = votes ^ (
                    torch.rand_like(votes, dtype=torch.float32) < flip_prob
                )

                # Majority vote
                maj = votes.float().mean(dim=1) > 0.5

                # Update ensemble judge stats
                TP_k[k_idx] += torch.sum(maj & correct_n[S_star_indices]).item()
                FN_k[k_idx] += torch.sum(~maj & correct_n[S_star_indices]).item()
                FP_k[k_idx] += torch.sum(maj & ~correct_n[S_star_indices]).item()
                TN_k[k_idx] += torch.sum(~maj & ~correct_n[S_star_indices]).item()

                # If S~ is empty after majority vote, skip
                if not maj.any():
                    continue

                # Update pipeline success count
                pipeline_success[n_idx, k_idx] += 1

                # Choose uniformly at random among majority-accepted answers
                chosen = S_star_indices[maj.float().multinomial(1).item()]

                # Check if chosen answer is a hallucination
                if not correct_n[chosen]:
                    hallucination_selection[n_idx, k_idx] += 1

    # normalise statistics
    empirical_p = empirical_p_count / (R * N_max)
    pipeline_failure = pipeline_failure / R
    hallucination_rate = hallucination_selection / pipeline_success.clamp_min(1e-9)

    q_pp = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else float("nan")
    q_np = fp_s / (fp_s + tn_s) if (fp_s + tn_s) > 0 else float("nan")

    # we keep track of Q++/Q-+ but don't use them further
    Q_pp = TP_k / (TP_k + FN_k).clamp_min(1e-9)
    Q_np = FP_k / (FP_k + TN_k).clamp_min(1e-9)

    judge_stats = {
        "single": {"q_pp": q_pp, "q_np": q_np},
        "ensemble": {"ks": odd_ks, "Q_pp": Q_pp.tolist(), "Q_np": Q_np.tolist()},
    }

    return (
        empirical_p,
        pipeline_failure,
        hallucination_rate,
        judge_stats,
    )


if __name__ == "__main__":
    # run full experiments for all tasks
    empirical_ps = []
    pipeline_failures = []
    hallucinations = []
    judge_stats_list = []

    for task in TASKS:
        empirical_p, pipe_fail, hall_sel, judge_stats = run_experiment(
            task, N_max=10, R=10000, k_max=17, flip_prob=0.25
        )

        empirical_ps.append(empirical_p)
        pipeline_failures.append(pipe_fail)
        hallucinations.append(hall_sel)
        judge_stats_list.append(judge_stats)

    # to torch tensors
    pipeline_failures = torch.stack(pipeline_failures)
    hallucinations = torch.stack(hallucinations)
    empirical_ps = torch.tensor(empirical_ps)

    # Plotting
    ks = torch.arange(1, hallucinations[0].shape[1] * 2, 2)
    N_values = torch.arange(1, pipeline_failures[0].shape[0] + 1)

    _cm = 1 / 2.54
    fig, axs = plt.subplots(
        2, len(pipeline_failures), figsize=(18 * _cm, 8 * _cm), sharey="row"
    )
    fig.subplots_adjust(wspace=0.15, hspace=0.60)

    tasknames = ["A", "B", "C"]
    labels = [["a)", "d)"], ["b)", "e)"], ["c)", "f)"]]

    for task_idx in range(len(pipeline_failures)):
        pf = pipeline_failures[task_idx]
        hall = hallucinations[task_idx]
        jstats = judge_stats_list[task_idx]
        p_est = empirical_ps[task_idx]

        lower_ci_pf, upper_ci_pf = compute_wilson_ci(pf, R=10000)

        q_pp = jstats["single"]["q_pp"]
        q_np = jstats["single"]["q_np"]

        # Theorem 2.1
        model_pf = (1 - (p_est * q_pp + (1 - p_est) * q_np)) ** N_values

        ax_pf = axs[0, task_idx]
        ax_hall = axs[1, task_idx]

        yerr_pf = torch.stack([pf - lower_ci_pf, upper_ci_pf - pf])
        eb = ax_pf.errorbar(
            N_values, pf, yerr=yerr_pf, fmt="o", color="C0", markersize=4, capsize=3
        )
        (line,) = ax_pf.plot(
            N_values, model_pf, color="C0", ls="--", linewidth=0.75, zorder=0
        )

        ax_pf.set_yscale("log")
        ax_pf.set_ylim(1e-3, 1)
        if task_idx == 0:
            ax_pf.set_ylabel("Pipeline failure rate")
            ax_pf.legend(
                [eb, line],
                ["Empirical", "Theorem 3.1"],
                fontsize=6,
                loc="lower left",
                ncol=1,
                bbox_to_anchor=(-0.025, -0.04),
                frameon=False,
            )

        ax_pf.text(
            0.5,
            1.1,
            f"Task {tasknames[task_idx]} ($p={p_est:.2f}$)",
            fontsize=10,
            transform=ax_pf.transAxes,
            ha="center",
            va="bottom",
        )
        ax_pf.set_title(f"{labels[task_idx][0]}", loc="left", fontsize=8, pad=4)
        ax_pf.set_xticks(N_values)
        ax_pf.set_xlabel("Repetitions $N$")

        # Hallucination rates are saved for each N but independent from N, average over them
        hallucination_rates = hall.mean(dim=0)
        lower_ci_h, upper_ci_h = compute_wilson_ci(hallucination_rates, R=10000)
        yerr_h = torch.stack(
            [hallucination_rates - lower_ci_h, upper_ci_h - hallucination_rates]
        )
        eb = ax_hall.errorbar(
            ks,
            hallucination_rates,
            yerr=yerr_h,
            fmt="o",
            markersize=4,
            color="C1",
            capsize=3,
        )

        Q_pp_list = []
        Q_np_list = []
        for k in ks:
            Q_pp_list.append(Q_majority_torch(q_pp, k).item())
            Q_np_list.append(Q_majority_torch(q_np, k).item())
        Q_pp_list = torch.tensor(Q_pp_list)
        Q_np_list = torch.tensor(Q_np_list)

        # Theorem 3.1
        model_hall = ((1 - p_est) * q_np * Q_np_list) / (
            p_est * q_pp * Q_pp_list + (1 - p_est) * q_np * Q_np_list
        )

        (line,) = ax_hall.plot(
            ks, model_hall, color="C1", ls="--", linewidth=0.75, zorder=0
        )

        if task_idx == 0:
            ax_hall.legend(
                [eb, line],
                ["Empirical", "Theorem 3.2"],
                fontsize=6,
                loc="lower left",
                ncol=1,
                bbox_to_anchor=(-0.025, -0.04),
                frameon=False,
            )

        ax_hall.set_yscale("log")
        ax_hall.set_ylim(1e-3, 1)
        ax_hall.set_xticks(ks)
        ax_hall.set_ylabel("Hallucination-selection rate") if task_idx == 0 else None
        ax_hall.set_xlabel("Judges $K$")
        ax_hall.set_title(f"{labels[task_idx][1]}", loc="left", fontsize=8, pad=4)

    fig.savefig("experiments.png", dpi=300, bbox_inches="tight")

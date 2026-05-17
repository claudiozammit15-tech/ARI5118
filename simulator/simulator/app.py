"""
6 pages in total:
- intro page
- one page for each of the four techniques
- one page where users can fine tune with multiple techniques under identical conditions and compare the results side by side.

Each technique page has 2 controls:
- dataset (one similar to original dataset, one very different)
- sample amount (to test with different dataset sizes)
Two universal controls on every page: dataset and sample count.

In each individual technique page, the user can check how much accuracy improved over the pretrained (which is the pretrained model + randomly initialised head for the new dataset)
This is more to get the user familiar with the techniques and the required configurations.

The comparison page is where the user can run multiple techniques under identical data conditions and compare their final accuracy side by side. 
This is where the user can really see the differences between techniques and get a more intuitive feel for which ones are more effective in which scenarios.

Evaluation metric is accuracy.

Run with:  streamlit run app.py
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from datasets import DATASETS, get_dataset, take_subset, get_dataset_info
from model_utils import (
    load_pretrained_resnet18
)
from training import (
    train_frozen_extraction, train_uniform_finetune,
    train_gradual_unfreezing, train_discriminative, train_lp_ft, evaluate,
)
from viz import plot_accuracy_bar, plot_training_curve, plot_comparison_bars


st.set_page_config(
    page_title="Transfer Learning Simulator",
    layout="wide",
)

@st.cache_resource
def load_dataset(name):
    return get_dataset(name)


@st.cache_resource
def load_model(num_classes):
    return load_pretrained_resnet18(num_classes)

def dataset_controls(key_prefix):
    """
    Dataset selector and sample count slider
    """
    col1, col2 = st.columns([2, 1])
    with col1:
        name = st.selectbox(
            "Dataset",
            list(DATASETS.keys()),
            key=f"{key_prefix}_ds",
            help=(
                "STL-10: natural photos similar to ImageNet (similar domain).\n"
                "EuroSAT: satellite imagery very different from ImageNet (different domain)."
                "MNIST: handwritten digits, nothing like ImageNet (extreme domain shift)."
            ),
        )
    with col2:
        n = st.slider(
            "Training samples",
            min_value=100, max_value=2000, value=500, step=100,
            key=f"{key_prefix}_n",
        )
    st.caption(get_dataset_info()[name])
    return name, n


def progress_cb(bar, text):
    def cb(epoch, total, loss, acc):
        bar.progress(epoch / total)
        text.text(f"Epoch {epoch}/{total}  -  loss {loss:.3f}  acc {acc:.1%}")
    return cb


def show_results(pre_acc, post_acc, technique, history):
    """Show the two key numbers and training curve."""
    # Metric cards
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Pretrained accuracy", f"{pre_acc:.1%}",
                  help="ResNet18 with its original ImageNet head removed "
                       "and a new random head attached. No fine-tuning.")
    with m2:
        delta_pts = (post_acc - pre_acc) * 100
        st.metric(f"After {technique}", f"{post_acc:.1%}",
                  delta=f"{delta_pts:+.1f} pts")
    with m3:
        st.metric("Improvement", f"{delta_pts:+.1f} pts",
                  delta_color="normal")

    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_accuracy_bar(pre_acc, post_acc, technique))
    with c2:
        st.pyplot(plot_training_curve(history, technique))


# side bar navigation

st.sidebar.title("🧠 Transfer Learning")
page = st.sidebar.radio(
    "Technique",
    [
        "Overview",
        "1 · Frozen extraction",
        "2 · End-to-end fine-tuning",
        "3 · Gradual unfreezing",
        "4 · Discriminative Learning Rates",
        "5 · Comparison",
    ],
)

# overview page
if page == "Overview":
    st.title("Transfer Learning and Fine-Tuning")
    st.markdown("""
This simulator lets you explore the four transfer learning and fine-tuning techniques from the study notes
by actually running them on a real ResNet18 and measuring the accuracy improvement, but most importantly, compare them against eachother in different scenarios.

**How to use it**

1. Pick a technique from the sidebar.
2. Choose a dataset:  STL-10 is similar to ImageNet; EuroSAT is drastically different; MNIST is extremely different.
3. Choose how many training samples to give the model.
4. Set the technique's hyperparameters.
5. Click **Train** and see how much accuracy improves over the pretrained baseline. (This is to understand how the technique works and what configurations it needs to be effective.)
6. Go to the **Comparison** page, select multiple techniques, set their hyperparameters, and train them under identical data conditions to see how they stack up against each other. (This is where you can really see the differences between techniques and get an intuitive feel for which ones are more effective in which scenarios.)

**The two numbers that matter**

Every page shows the same comparison:
- **Pretrained accuracy** - what ResNet18 gets with a new random head and no fine-tuning at all.
- **Fine-tuned accuracy** - what it gets after applying the chosen technique.

These two numbers help you understand how the technique works. The Comparison page is what is most interesting important.

**Datasets**

| Dataset | Domain | Classes | Size |
|---------|--------|---------|------|
| STL-10 | Natural photos (similar to ImageNet) | 10 | 5,000 train |
| EuroSAT | Satellite imagery (very different) | 10 | ~21,600 train |
| MNIST | Handwritten digits (extremely different) | 10 | 60,000 train |

STL-10 transfers easily - even the pretrained baseline will score reasonably well.
EuroSAT is the harder case - the baseline will be lower and fine-tuning matters more.
MNIST is the most extreme case - the baseline will be quite low and fine-tuning is crucial for good performance.
    """)


#Page 1- Frozen extraction

elif page == "1 · Frozen extraction":
    st.title("Frozen feature extraction")
    st.markdown(
        "Freeze the entire backbone. Train only the new classification head. "
        "The backbone's ImageNet weights are never updated - the head learns "
        "to classify using the pretrained features as-is."
    )

    dataset_name, n_samples = dataset_controls("fe")
    st.markdown("---")

    col_hp, col_out = st.columns([1, 2])
    with col_hp:
        st.subheader("Hyperparameters")
        lr     = st.select_slider("Head learning rate",
                    [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
                    value=1e-2, format_func=lambda x: f"{x:.0e}")
        epochs = st.slider("Epochs", 3, 30, 10)
        seed   = st.number_input("Seed", 0, 999, 0)
        btn    = st.button("Train frozen probe", type="primary")

    with col_out:
        if btn:
            train_pool, test_pool, class_names = load_dataset(dataset_name)
            train_sub = take_subset(train_pool, n_samples, seed=seed)
            base = load_model(len(class_names))

            with st.spinner("Evaluating pretrained baseline..."):
                pre = evaluate(base, test_pool, max_samples=500)

            bar, txt = st.progress(0), st.empty()
            trained, hist = train_frozen_extraction(
                base, train_sub, lr=lr, epochs=epochs, seed=seed,
                progress_callback=progress_cb(bar, txt),
            )
            bar.empty(); txt.empty()

            with st.spinner("Evaluating fine-tuned model..."):
                post = evaluate(trained, test_pool, max_samples=500)

            show_results(pre, post, "Frozen extraction", hist)

            with st.expander("What happened?"):
                st.markdown(
                    f"Only the head was trained - "                   
                    "The backbone weights were not touched. "
                    "Whatever accuracy improvement you see came entirely from "
                    "learning better decision boundaries in the frozen feature space."
                )
        else:
            st.info("Configure hyperparameters and click **Train frozen probe**.")


# Page 2End-to-end fine-tuning

elif page == "2 · End-to-end fine-tuning":
    st.title("End-to-end fine-tuning")
    st.markdown(
        "All parameters are trainable. Two end-to-end strategies: "
        "**Uniform fine-tuning** (one learning rate for every layer) and "
        "**LP-FT** (train the head first to stabilise it, then fine-tune everything)."
    )

    dataset_name, n_samples = dataset_controls("e2e")
    st.markdown("---")

    tab_uni, tab_lpft = st.tabs(["Uniform fine-tuning", "LP-FT"])

    with tab_uni:
        col_hp, col_out = st.columns([1, 2])
        with col_hp:
            st.subheader("Hyperparameters")
            uni_lr  = st.select_slider("Learning rate (all layers)",
                        [1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 1e-2],
                        value=1e-4, format_func=lambda x: f"{x:.0e}",
                        key="uni_lr")
            uni_ep  = st.slider("Epochs", 2, 15, 5, key="uni_ep")
            uni_seed = st.number_input("Seed", 0, 999, 0, key="uni_seed")
            uni_btn = st.button("Train uniform FT", type="primary", key="uni_btn")

        with col_out:
            if uni_btn:
                train_pool, test_pool, class_names = load_dataset(dataset_name)
                train_sub = take_subset(train_pool, n_samples, seed=uni_seed)
                base = load_model(len(class_names))

                with st.spinner("Evaluating pretrained baseline..."):
                    pre = evaluate(base, test_pool, max_samples=500)

                bar, txt = st.progress(0), st.empty()
                trained, hist = train_uniform_finetune(
                    base, train_sub, lr=uni_lr, epochs=uni_ep, seed=uni_seed,
                    progress_callback=progress_cb(bar, txt),
                )
                bar.empty(); txt.empty()

                with st.spinner("Evaluating..."):
                    post = evaluate(trained, test_pool, max_samples=500)

                show_results(pre, post, "Uniform FT", hist)

                if uni_lr >= 1e-2:
                    st.warning(
                        "LR ≥ 1e-2 is in the catastrophic-forgetting zone. "
                        "Large gradients from the random head are likely "
                        "overwriting pretrained features before they can adapt."
                    )
            else:
                st.info("Configure and click **Train uniform FT**.")

    with tab_lpft:
        col_hp, col_out = st.columns([1, 2])
        with col_hp:
            st.subheader("Stage 1: Linear probe (only head trained)")
            lp_lr = st.select_slider("Head LR",
                        [1e-3, 5e-3, 1e-2, 5e-2],
                        value=1e-2, format_func=lambda x: f"{x:.0e}",
                        key="lp_lr")
            lp_ep = st.slider("Epochs", 2, 10, 5, key="lp_ep")
            st.subheader("Stage 2: Full fine-tune")
            ft_lr = st.select_slider("LR (all layers including head)",
                        [1e-6, 1e-5, 1e-4, 1e-3],
                        value=1e-4, format_func=lambda x: f"{x:.0e}",
                        key="ft_lr")
            ft_ep = st.slider("Epochs", 2, 10, 3, key="ft_ep")
            lpft_seed = st.number_input("Seed", 0, 999, 0, key="lpft_seed")
            lpft_btn = st.button("Train LP-FT", type="primary", key="lpft_btn")

        with col_out:
            if lpft_btn:
                train_pool, test_pool, class_names = load_dataset(dataset_name)
                train_sub = take_subset(train_pool, n_samples, seed=lpft_seed)
                base = load_model(len(class_names))

                with st.spinner("Evaluating pretrained baseline..."):
                    pre = evaluate(base, test_pool, max_samples=500)

                bar, txt = st.progress(0), st.empty()
                trained, hist = train_lp_ft(
                    base, train_sub,
                    lp_lr=lp_lr, lp_epochs=lp_ep,
                    ft_lr=ft_lr, ft_epochs=ft_ep,
                    seed=lpft_seed,
                    progress_callback=progress_cb(bar, txt),
                )
                bar.empty(); txt.empty()

                with st.spinner("Evaluating..."):
                    post = evaluate(trained, test_pool, max_samples=500)

                show_results(pre, post, "LP-FT", hist)
                st.caption(
                    "The dashed vertical line on the training curve marks where "
                    "Stage 1 (head only) ended and Stage 2 (full fine-tune) began."
                )
            else:
                st.info("Configure both stages and click **Train LP-FT**.")


# Page 3: Gradual unfreezing

elif page == "3 · Gradual unfreezing":
    st.title("Gradual unfreezing")
    st.markdown(
        "Start with only the head trainable. After each stage, unfreeze one "
        "more backbone block working backward from output toward input. "
        "Five stages total: head → +layer4 → +layer3 → +layer2 → +layer1+stem."
    )

    dataset_name, n_samples = dataset_controls("gu")
    st.markdown("---")

    col_hp, col_out = st.columns([1, 2])
    with col_hp:
        st.subheader("Hyperparameters")
        lr  = st.select_slider("Learning rate (constant throughout)",
                [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
                value=1e-4, format_func=lambda x: f"{x:.0e}")
        eps = st.slider("Epochs per stage", 1, 5, 2,
                help="Total training = 5 stages × this value.")
        st.caption(f"Total epochs: {5 * eps}")
        seed = st.number_input("Seed", 0, 999, 0, key="gu_seed")
        btn  = st.button("Train gradual unfreezing", type="primary")

    with col_out:
        if btn:
            train_pool, test_pool, class_names = load_dataset(dataset_name)
            train_sub = take_subset(train_pool, n_samples, seed=seed)
            base = load_model(len(class_names))

            with st.spinner("Evaluating pretrained baseline..."):
                pre = evaluate(base, test_pool, max_samples=500)

            bar, txt = st.progress(0), st.empty()
            trained, hist = train_gradual_unfreezing(
                base, train_sub, lr=lr, epochs_per_stage=eps, seed=seed,
                progress_callback=progress_cb(bar, txt),
            )
            bar.empty(); txt.empty()

            with st.spinner("Evaluating..."):
                post = evaluate(trained, test_pool, max_samples=500)

            show_results(pre, post, "Gradual unfreezing", hist)
            st.caption(
                "Dashed lines on the training curve mark stage transitions. "
                "Accuracy often jumps slightly at each transition as a newly "
                "unfrozen layer begins adapting."
            )
        else:
            st.info("Configure and click **Train gradual unfreezing**.")


# Page 4: Discriminative Learning Rates

elif page == "4 · Discriminative Learning Rates":
    st.title("Discriminative learning rates")
    st.markdown(
        "All layers train simultaneously, but each group has its own learning rate. "
        "Layers closer to the input which encode more general features"
        "change less. Layers closer to the output which need the most "
        "task-specific adaptation change more."
    )

    dataset_name, n_samples = dataset_controls("dlr")
    st.markdown("---")

    col_hp, col_out = st.columns([1, 2])
    with col_hp:
        st.subheader("Learning rate per group")
        st.caption("Convention: head highest, early backbone lowest.")
        lr_h = st.select_slider("Head (fc)",
                [1e-4, 1e-3, 1e-2, 5e-2],
                value=1e-2, format_func=lambda x: f"{x:.0e}", key="dlr_h")
        lr_l = st.select_slider("Late backbone (layer4)",
                [1e-5, 1e-4, 1e-3, 1e-2],
                value=1e-3, format_func=lambda x: f"{x:.0e}", key="dlr_l")
        lr_m = st.select_slider("Mid backbone (layer2 + layer3)",
                [1e-6, 1e-5, 1e-4, 1e-3],
                value=1e-4, format_func=lambda x: f"{x:.0e}", key="dlr_m")
        lr_e = st.select_slider("Early backbone (stem + layer1)",
                [1e-7, 1e-6, 1e-5, 1e-4],
                value=1e-5, format_func=lambda x: f"{x:.0e}", key="dlr_e")
        epochs = st.slider("Epochs", 2, 15, 5, key="dlr_ep")
        seed   = st.number_input("Seed", 0, 999, 0, key="dlr_seed")
        btn    = st.button("Train discriminative LRs", type="primary")

    with col_out:
        if btn:
            train_pool, test_pool, class_names = load_dataset(dataset_name)
            train_sub = take_subset(train_pool, n_samples, seed=seed)
            base = load_model(len(class_names))

            with st.spinner("Evaluating pretrained baseline..."):
                pre = evaluate(base, test_pool, max_samples=500)

            bar, txt = st.progress(0), st.empty()
            trained, hist = train_discriminative(
                base, train_sub,
                lr_head=lr_h, lr_late=lr_l, lr_mid=lr_m, lr_early=lr_e,
                epochs=epochs, seed=seed,
                progress_callback=progress_cb(bar, txt),
            )
            bar.empty(); txt.empty()

            with st.spinner("Evaluating..."):
                post = evaluate(trained, test_pool, max_samples=500)

            show_results(pre, post, "Discriminative LRs", hist)

            # Show the LR ratios so the principle is tangible
            with st.expander("LR ratios across groups"):
                st.markdown(
                    f"| Group | LR | Ratio to head |\n"
                    f"|---|---|---|\n"
                    f"| Head | `{lr_h:.0e}` | 1× |\n"
                    f"| Late backbone | `{lr_l:.0e}` | {lr_h/lr_l:.0f}× lower |\n"
                    f"| Mid backbone | `{lr_m:.0e}` | {lr_h/lr_m:.0f}× lower |\n"
                    f"| Early backbone | `{lr_e:.0e}` | {lr_h/lr_e:.0f}× lower |"
                )
        else:
            st.info("Configure learning rates and click **Train discriminative LRs**.")


# Page 5: Comparison 

elif page == "5 · Comparison":
    st.title("Strategy comparison")
    st.markdown(
        "Train two or more strategies under identical data conditions "
        "and compare their final accuracy side by side."
    )

    dataset_name, n_samples = dataset_controls("cmp")
    seed = st.number_input("Seed (shared across all strategies)", 0, 999, 0,
                            key="cmp_seed")
    st.markdown("---")

    st.subheader("Select strategies to compare")
    strategies = st.multiselect(
        "Strategies",
        ["Frozen extraction", "Uniform FT", "LP-FT",
         "Gradual unfreezing", "Discriminative LRs"],
        default=["Frozen extraction", "Uniform FT"],
    )

    if not strategies:
        st.warning("Select at least one strategy.")
        st.stop()

    configs = {}

    if "Frozen extraction" in strategies:
        with st.expander("Frozen extraction - hyperparameters", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                lr = st.select_slider("Head LR",
                        [1e-4, 1e-3, 1e-2, 5e-2], value=1e-2,
                        format_func=lambda x: f"{x:.0e}", key="cmp_fe_lr")
            with c2:
                ep = st.slider("Epochs", 3, 20, 8, key="cmp_fe_ep")
            configs["Frozen extraction"] = {"lr": lr, "epochs": ep}

    if "Uniform FT" in strategies:
        with st.expander("Uniform FT - hyperparameters", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                lr = st.select_slider("LR (all)",
                        [1e-6, 1e-5, 1e-4, 1e-3], value=1e-4,
                        format_func=lambda x: f"{x:.0e}", key="cmp_uni_lr")
            with c2:
                ep = st.slider("Epochs", 2, 10, 5, key="cmp_uni_ep")
            configs["Uniform FT"] = {"lr": lr, "epochs": ep}

    if "LP-FT" in strategies:
        with st.expander("LP-FT - hyperparameters", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                lp_lr = st.select_slider("LP LR",
                            [1e-3, 1e-2, 5e-2], value=1e-2,
                            format_func=lambda x: f"{x:.0e}", key="cmp_lp_lr")
            with c2:
                lp_ep = st.slider("LP epochs", 2, 8, 4, key="cmp_lp_ep")
            with c3:
                ft_lr = st.select_slider("FT LR",
                            [1e-6, 1e-5, 1e-4, 1e-3], value=1e-4,
                            format_func=lambda x: f"{x:.0e}", key="cmp_ft_lr")
            with c4:
                ft_ep = st.slider("FT epochs", 2, 8, 3, key="cmp_ft_ep")
            configs["LP-FT"] = {"lp_lr": lp_lr, "lp_epochs": lp_ep,
                                  "ft_lr": ft_lr, "ft_epochs": ft_ep}

    if "Gradual unfreezing" in strategies:
        with st.expander("Gradual unfreezing - hyperparameters", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                lr = st.select_slider("LR",
                        [1e-5, 1e-4, 1e-3], value=1e-4,
                        format_func=lambda x: f"{x:.0e}", key="cmp_gu_lr")
            with c2:
                eps = st.slider("Epochs per stage", 1, 4, 2, key="cmp_gu_ep")
            configs["Gradual unfreezing"] = {"lr": lr, "epochs_per_stage": eps}

    if "Discriminative LRs" in strategies:
        with st.expander("Discriminative LRs - hyperparameters", expanded=True):
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                lh = st.select_slider("Head", [1e-3, 1e-2, 5e-2], value=1e-2,
                        format_func=lambda x: f"{x:.0e}", key="cmp_dlr_h")
            with c2:
                ll = st.select_slider("Late", [1e-5, 1e-4, 1e-3], value=1e-3,
                        format_func=lambda x: f"{x:.0e}", key="cmp_dlr_l")
            with c3:
                lm = st.select_slider("Mid", [1e-6, 1e-5, 1e-4], value=1e-4,
                        format_func=lambda x: f"{x:.0e}", key="cmp_dlr_m")
            with c4:
                le = st.select_slider("Early", [1e-6, 1e-5, 1e-4], value=1e-5,
                        format_func=lambda x: f"{x:.0e}", key="cmp_dlr_e")
            with c5:
                ep = st.slider("Epochs", 2, 10, 5, key="cmp_dlr_ep")
            configs["Discriminative LRs"] = {
                "lr_head": lh, "lr_late": ll, "lr_mid": lm, "lr_early": le,
                "epochs": ep,
            }

    st.markdown("---")
    run_btn = st.button("Train all selected strategies", type="primary")

    if run_btn:
        train_pool, test_pool, class_names = load_dataset(dataset_name)
        train_sub = take_subset(train_pool, n_samples, seed=seed)
        base = load_model(len(class_names))

        with st.spinner("Evaluating pretrained baseline..."):
            pre = evaluate(base, test_pool, max_samples=500)
        st.info(f"Pretrained baseline: **{pre:.1%}**")

        results = []

        def run_strategy(name, cfg):
            bar, txt = st.progress(0), st.empty()
            cb = progress_cb(bar, txt)

            if name == "Frozen extraction":
                trained, _ = train_frozen_extraction(
                    base, train_sub, lr=cfg["lr"], epochs=cfg["epochs"],
                    seed=seed, progress_callback=cb)
            elif name == "Uniform FT":
                trained, _ = train_uniform_finetune(
                    base, train_sub, lr=cfg["lr"], epochs=cfg["epochs"],
                    seed=seed, progress_callback=cb)
            elif name == "LP-FT":
                trained, _ = train_lp_ft(
                    base, train_sub,
                    lp_lr=cfg["lp_lr"], lp_epochs=cfg["lp_epochs"],
                    ft_lr=cfg["ft_lr"], ft_epochs=cfg["ft_epochs"],
                    seed=seed, progress_callback=cb)
            elif name == "Gradual unfreezing":
                trained, _ = train_gradual_unfreezing(
                    base, train_sub, lr=cfg["lr"],
                    epochs_per_stage=cfg["epochs_per_stage"],
                    seed=seed, progress_callback=cb)
            elif name == "Discriminative LRs":
                trained, _ = train_discriminative(
                    base, train_sub,
                    lr_head=cfg["lr_head"], lr_late=cfg["lr_late"],
                    lr_mid=cfg["lr_mid"], lr_early=cfg["lr_early"],
                    epochs=cfg["epochs"], seed=seed, progress_callback=cb)

            bar.empty(); txt.empty()
            acc = evaluate(trained, test_pool, max_samples=500)
            return acc

        for name in strategies:
            st.markdown(f"**Training: {name}**")
            acc = run_strategy(name, configs[name])
            st.success(f"{name}: **{acc:.1%}**  ({(acc-pre)*100:+.1f} pts vs pretrained)")
            results.append((name, pre, acc))

        st.markdown("---")
        st.subheader("Final comparison")
        st.pyplot(plot_comparison_bars(results))

"""
Streamlit dashboard for the Customer Churn Prediction project.

Run:  streamlit run app.py

The dashboard never trains a model. It reads:
  - data/processed/telco_churn_clean.csv      (cleaned dataset)
  - reports/*.csv, reports/*.json, figures    (results written by src/train_model.py)
  - models/churn_model.joblib                 (saved pipeline used for predictions)
"""

import json

import altair as alt
import pandas as pd
import streamlit as st

from src import config
from src.data_preprocessing import load_clean_data, tenure_group
from src.predict import InvalidCustomerInput, load_model, predict_customers

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")

ORANGE, BLUE = "#DD8452", "#4C72B0"


# ----------------------------------------------------------------------------
# Cached loaders (read once per session; nothing is retrained here)
# ----------------------------------------------------------------------------
@st.cache_data
def get_data() -> pd.DataFrame:
    if config.PROCESSED_DATA_PATH.exists():
        df = pd.read_csv(config.PROCESSED_DATA_PATH)
    else:
        df = load_clean_data()
    df["ChurnLabel"] = df[config.TARGET].map({0: "No", 1: "Yes"})
    df["TenureGroup"] = tenure_group(df["tenure"]).astype(str)
    return df


@st.cache_data
def get_json(path):
    with open(path) as f:
        return json.load(f)


@st.cache_data
def get_csv(path):
    return pd.read_csv(path)


@st.cache_resource
def get_model():
    return load_model()


def artifacts_ready() -> bool:
    needed = [config.MODEL_PATH, config.MODEL_METADATA_PATH, config.MODEL_COMPARISON_PATH,
              config.EDA_SUMMARY_PATH, config.CONFUSION_MATRICES_PATH]
    missing = [p.relative_to(config.PROJECT_ROOT) for p in needed if not p.exists()]
    if missing:
        st.error("Missing generated files: " + ", ".join(map(str, missing)) +
                 ".\n\nRun `python -m src.eda` and `python -m src.train_model` first.")
        return False
    return True


def churn_rate_chart(df: pd.DataFrame, col: str, sort=None, title=None):
    g = (df.groupby(col, observed=True)[config.TARGET]
           .agg(customers="count", churn_rate="mean").reset_index())
    g["churn_rate_pct"] = (100 * g["churn_rate"]).round(1)
    bars = alt.Chart(g).mark_bar(color=ORANGE).encode(
        x=alt.X(f"{col}:N", sort=sort, title=col),
        y=alt.Y("churn_rate_pct:Q", title="Churn rate (%)"),
        tooltip=[col, "customers", alt.Tooltip("churn_rate_pct", title="churn %")],
    )
    text = bars.mark_text(dy=-6).encode(text=alt.Text("churn_rate_pct:Q", format=".1f"))
    rule = alt.Chart(pd.DataFrame({"y": [100 * df[config.TARGET].mean()]})).mark_rule(
        strokeDash=[4, 4], color="grey").encode(y="y:Q")
    return (bars + text + rule).properties(title=title or f"Churn rate by {col}", height=320)


def rate_lookup(summary, col, level):
    for r in summary["churn_rate_by"][col]:
        if r[col] == level:
            return r["churn_rate_%"], r["customers"]
    return None, None


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
def page_overview(df, summary, meta):
    st.title("Customer Churn Prediction and Retention Analysis")
    st.caption("Predictive Analytics (CSE3141) mini-project · Manipal University Jaipur")
    st.markdown(
        "**Objective:** analyse historical telecom customer data, identify patterns "
        "associated with churn, and build classification models that estimate whether "
        "a customer is likely to leave."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total customers", f"{summary['n_customers']:,}")
    c2.metric("Churned customers", f"{summary['n_churned']:,}")
    c3.metric("Historical churn rate", f"{summary['churn_rate']:.1%}")
    c4.metric("Model features", len(meta["features"]))

    st.subheader("Dataset summary")
    dq = summary["data_quality_raw"]
    st.markdown(
        f"""
- **Source:** IBM *Telco Customer Churn* sample dataset (file `{meta['dataset_file']}`).
- **Size:** {dq['n_rows']:,} rows × {dq['n_columns']} columns; one row per customer.
- **Target:** `Churn` - customer left within the last month (Yes = 1, No = 0).
- **Excluded:** `customerID` (a unique label with no predictive meaning).
- **Data quality:** {sum(dq['missing_values (NaN)'].values()) if dq['missing_values (NaN)'] else 0} NaN values,
  {dq['blank_strings'].get('TotalCharges', 0)} blank `TotalCharges` strings (customers with tenure 0, set to 0),
  {dq['duplicate_rows']} duplicate rows, {dq['duplicate_customer_ids']} duplicate customer IDs.
"""
    )
    st.subheader("Sample of the cleaned data")
    st.dataframe(df.drop(columns=["ChurnLabel", "TenureGroup"]).head(10), width="stretch")

    st.subheader("Descriptive statistics (numeric features)")
    st.dataframe(pd.DataFrame(summary["descriptive_stats"]).T.round(2), width="stretch")


def page_eda(df, summary):
    st.title("Exploratory Data Analysis")
    tab1, tab2, tab3, tab4 = st.tabs(["Churn & contract", "Tenure & charges", "Services & billing",
                                      "Correlation & tests"])

    with tab1:
        c1, c2 = st.columns(2)
        counts = df["ChurnLabel"].value_counts().rename_axis("Churn").reset_index(name="customers")
        c1.altair_chart(
            alt.Chart(counts).mark_arc(innerRadius=60).encode(
                theta="customers:Q",
                color=alt.Color("Churn:N", scale=alt.Scale(domain=["No", "Yes"], range=[BLUE, ORANGE])),
                tooltip=["Churn", "customers"],
            ).properties(title="Churn distribution", height=320),
            use_container_width=True,
        )
        c2.altair_chart(churn_rate_chart(df, "Contract", ["Month-to-month", "One year", "Two year"]),
                        use_container_width=True)
        st.image(str(config.FIGURES_DIR / "eda_tenure_contract_heatmap.png"), width=650,
                 caption="Churn rate by tenure group and contract type")

    with tab2:
        c1, c2 = st.columns(2)
        c1.altair_chart(churn_rate_chart(df, "TenureGroup", ["0-12 m", "13-24 m", "25-48 m", "49-72 m"]),
                        use_container_width=True)
        num = c2.selectbox("Numeric feature", config.NUMERIC_FEATURES, index=1)
        c2.altair_chart(
            alt.Chart(df).transform_density(num, groupby=["ChurnLabel"], as_=[num, "density"])
            .mark_area(opacity=0.5).encode(
                x=f"{num}:Q", y="density:Q",
                color=alt.Color("ChurnLabel:N", title="Churn",
                                scale=alt.Scale(domain=["No", "Yes"], range=[BLUE, ORANGE])),
            ).properties(title=f"{num} density by churn", height=320),
            use_container_width=True,
        )
        st.altair_chart(
            alt.Chart(df.sample(min(2500, len(df)), random_state=0)).mark_circle(size=18, opacity=0.5).encode(
                x="tenure:Q", y="MonthlyCharges:Q",
                color=alt.Color("ChurnLabel:N", title="Churn",
                                scale=alt.Scale(domain=["No", "Yes"], range=[BLUE, ORANGE])),
                tooltip=["tenure", "MonthlyCharges", "Contract", "ChurnLabel"],
            ).properties(title="Tenure vs monthly charges (random sample of 2,500 customers)", height=380)
            .interactive(),
            use_container_width=True,
        )
        st.image(str(config.FIGURES_DIR / "eda_boxplots.png"), caption="Box plots (IQR outlier check)")
        st.dataframe(pd.DataFrame(summary["iqr_outliers"]).round(2), width="stretch")

    with tab3:
        options = [c for c in config.CATEGORICAL_FEATURES if c != "Contract"]
        col = st.selectbox("Attribute", options, index=options.index("InternetService"))
        st.altair_chart(churn_rate_chart(df, col), use_container_width=True)
        st.caption("Dashed line = overall churn rate. Add-on columns: 'No' includes customers "
                   "without internet service.")

    with tab4:
        st.image(str(config.FIGURES_DIR / "eda_correlation_heatmap.png"), width=600)
        st.markdown("**Chi-square test of independence (categorical feature vs churn)**")
        chi = pd.DataFrame(summary["chi_square"])
        chi["significant (p<0.05)"] = chi["p_value"] < 0.05
        st.dataframe(chi.round(4), width="stretch")
        st.caption("Cramér's V measures association strength (0 = none, 1 = perfect). "
                   "Association is not causation.")


def page_models(meta):
    st.title("Model Evaluation")
    comp = get_csv(config.MODEL_COMPARISON_PATH)
    st.markdown(
        f"Data was split **{meta['n_train']:,} train / {meta['n_test']:,} test** (stratified, "
        f"seed {meta['random_state']}). All model selection used **5-fold stratified cross-validation "
        "on the training set only**; the test set was used once, for the final numbers below."
    )

    show = comp[["model", "stage", "cv_roc_auc_mean", "cv_roc_auc_std", "test_accuracy",
                 "test_precision", "test_recall", "test_f1", "test_roc_auc", "selected"]].copy()
    show.columns = ["Model", "Stage", "CV ROC-AUC", "CV std", "Accuracy", "Precision", "Recall",
                    "F1", "ROC-AUC", "Selected"]
    st.subheader("Model comparison (held-out test set)")
    st.dataframe(show.style.format(precision=4).highlight_max(
        subset=["CV ROC-AUC", "Recall", "F1", "ROC-AUC"], color="#d9f2e3"), width="stretch")

    st.subheader(f"Selected model: {meta['model_name']}")
    tm = meta["test_metrics"]
    c = st.columns(5)
    for col, (k, label) in zip(c, [("accuracy", "Accuracy"), ("precision", "Precision"),
                                   ("recall", "Recall"), ("f1", "F1-score"), ("roc_auc", "ROC-AUC")]):
        col.metric(label, f"{tm[k]:.3f}")
    st.markdown(
        f"- **Selection rule:** {meta['selection_rule']}.\n"
        f"- **Best hyperparameters:** `{meta['best_params']}`\n"
        f"- **Decision threshold:** {meta['decision_threshold']}"
    )
    tuned = comp[comp.stage == "tuned"].sort_values("cv_roc_auc_mean", ascending=False)
    gap = tuned.iloc[0]["cv_roc_auc_mean"] - tuned.iloc[1]["cv_roc_auc_mean"]
    st.info(
        f"The runner-up ({tuned.iloc[1]['model']}) is only {gap:.4f} CV ROC-AUC behind, which is smaller "
        f"than one cross-validation standard deviation (~{tuned.iloc[0]['cv_roc_auc_std']:.3f}). "
        "The models perform very similarly; Logistic Regression remains the most interpretable."
    )
    baseline_acc = comp.loc[comp.model == "Majority-class baseline", "test_accuracy"].iloc[0]
    st.warning(
        f"**Why accuracy alone misleads:** a model that always predicts 'No churn' scores "
        f"{baseline_acc:.1%} accuracy on this test set but catches 0 churners (recall = 0)."
    )

    st.subheader("Confusion matrix")
    cms = get_json(config.CONFUSION_MATRICES_PATH)
    key = st.selectbox("Model", list(cms.keys()), index=list(cms.keys()).index(f"{meta['model_name']} (tuned)"))
    cm = cms[key]
    cm_df = pd.DataFrame([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
                         index=["Actual: No churn", "Actual: Churn"],
                         columns=["Predicted: No churn", "Predicted: Churn"])
    c1, c2 = st.columns([1, 1])
    c1.dataframe(cm_df, width="stretch")
    c2.markdown(
        f"- **True positives:** {cm['tp']} churners correctly flagged\n"
        f"- **False negatives:** {cm['fn']} churners missed\n"
        f"- **False positives:** {cm['fp']} loyal customers flagged unnecessarily\n"
        f"- **True negatives:** {cm['tn']} loyal customers correctly cleared"
    )

    st.subheader("Evaluation plots")
    st.image(str(config.FIGURES_DIR / "model_roc_pr_curves.png"))
    st.image(str(config.FIGURES_DIR / "model_metric_comparison.png"))
    c1, c2 = st.columns(2)
    c1.image(str(config.FIGURES_DIR / "model_permutation_importance.png"))
    c2.image(str(config.FIGURES_DIR / "model_validation_curve_tree_depth.png"))

    with st.expander("Logistic Regression odds ratios"):
        odds = get_csv(config.ODDS_RATIOS_PATH)
        st.dataframe(odds.round(4), width="stretch")
        st.markdown(
            "An odds ratio > 1 means higher churn odds than the reference category (or per +1 SD "
            "for numeric features), holding other features constant. MonthlyCharges is almost fully "
            "determined by the services a customer has, so its coefficient and those of the service "
            "columns partly offset each other - interpret them together, not one by one."
        )
    with st.expander("Feature-set ablation (training CV)"):
        st.dataframe(get_csv(config.FEATURE_ABLATION_PATH).round(4), width="stretch")


def page_predict(meta):
    st.title("Customer Churn Prediction")
    st.markdown(f"Uses the saved **{meta['model_name']}** pipeline (`models/churn_model.joblib`). "
                "No retraining happens here.")
    cats = config.ALLOWED_CATEGORIES

    with st.form("customer"):
        st.markdown("**Demographics**")
        c = st.columns(4)
        gender = c[0].selectbox("Gender", cats["gender"])
        senior = c[1].selectbox("Senior citizen", cats["SeniorCitizen"])
        partner = c[2].selectbox("Partner", cats["Partner"])
        dependents = c[3].selectbox("Dependents", cats["Dependents"])

        st.markdown("**Account**")
        c = st.columns(4)
        tenure = c[0].number_input("Tenure (months)", 0, 72, 12, help="Dataset range is 0-72 months")
        contract = c[1].selectbox("Contract", cats["Contract"])
        paperless = c[2].selectbox("Paperless billing", cats["PaperlessBilling"])
        payment = c[3].selectbox("Payment method", cats["PaymentMethod"],
                                 index=cats["PaymentMethod"].index("Electronic check"))
        c = st.columns(4)
        monthly = c[0].number_input("Monthly charges ($)", 18.0, 120.0, 70.0, step=0.5,
                                    help="Dataset range is 18.25-118.75")
        auto_total = c[1].checkbox("Estimate total charges as tenure × monthly", value=True)
        total_in = c[2].number_input("Total charges ($)", 0.0, 9000.0, 840.0, step=10.0,
                                     help="Used only if the estimate box is unticked")

        st.markdown("**Services**")
        c = st.columns(4)
        phone = c[0].selectbox("Phone service", cats["PhoneService"], index=1)
        multi = c[1].selectbox("Multiple lines", cats["MultipleLines"])
        internet = c[2].selectbox("Internet service", cats["InternetService"], index=1)
        c = st.columns(6)
        addons = {}
        for i, a in enumerate(config.INTERNET_ADDONS):
            addons[a] = c[i].selectbox(a, cats[a])
        submitted = st.form_submit_button("Predict churn", type="primary")

    if submitted:
        record = {
            "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone, "MultipleLines": multi,
            "InternetService": internet, "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "MonthlyCharges": monthly,
            "TotalCharges": tenure * monthly if auto_total else total_in, **addons,
        }
        try:
            get_model()
            res = predict_customers(record, threshold=meta["decision_threshold"]).iloc[0]
        except InvalidCustomerInput as e:
            st.error(f"Invalid input: {e}")
            return
        p = res["churn_probability"]
        c1, c2 = st.columns([1, 2])
        if res["prediction"] == "Churn":
            c1.error(f"### Prediction: Churn\nEstimated probability: **{p:.1%}**")
        else:
            c1.success(f"### Prediction: No Churn\nEstimated probability of churn: **{p:.1%}**")
        c2.progress(float(p), text=f"Churn probability {p:.1%} (threshold {meta['decision_threshold']:.0%})")
        c2.caption(
            "This is a statistical estimate from patterns in historical data, not a guarantee. "
            "The model was trained with class weighting, so probabilities are shifted upward "
            "relative to the true base rate and are best used for ranking customers by risk. "
            f"Test-set precision of this model is {meta['test_metrics']['precision']:.0%}: "
            "many customers flagged as 'Churn' would in fact stay."
        )
        with st.expander("Input sent to the model"):
            st.json(record)


def page_retention(df, summary):
    st.title("Retention Recommendations")
    st.markdown(
        "Section A lists **data-driven findings** computed from this dataset. Section B gives "
        "**general recommendations** inspired by those findings. The findings are statistical "
        "associations; they do not prove that changing a factor would reduce churn. Any action "
        "should be tested (e.g. with a controlled A/B experiment) before roll-out."
    )
    overall = 100 * summary["churn_rate"]
    m2m, n_m2m = rate_lookup(summary, "Contract", "Month-to-month")
    two, _ = rate_lookup(summary, "Contract", "Two year")
    new, n_new = rate_lookup(summary, "TenureGroup", "0-12 m")
    loyal, _ = rate_lookup(summary, "TenureGroup", "49-72 m")
    echk, n_echk = rate_lookup(summary, "PaymentMethod", "Electronic check")
    card, _ = rate_lookup(summary, "PaymentMethod", "Credit card (automatic)")
    fiber, n_fiber = rate_lookup(summary, "InternetService", "Fiber optic")
    dsl, _ = rate_lookup(summary, "InternetService", "DSL")
    sec_no, _ = rate_lookup(summary, "OnlineSecurity", "No")
    sec_yes, _ = rate_lookup(summary, "OnlineSecurity", "Yes")
    ts_no, _ = rate_lookup(summary, "TechSupport", "No")
    ts_yes, _ = rate_lookup(summary, "TechSupport", "Yes")
    sen, _ = rate_lookup(summary, "SeniorCitizen", "Yes")
    pl, _ = rate_lookup(summary, "PaperlessBilling", "Yes")

    st.header("A. Data-driven findings")
    st.markdown(f"Overall churn rate: **{overall:.1f}%**")
    findings = pd.DataFrame([
        ["Contract", f"Month-to-month: {m2m:.1f}% ({n_m2m:,} customers)", f"Two year: {two:.1f}%"],
        ["Tenure", f"0-12 months: {new:.1f}% ({n_new:,} customers)", f"49-72 months: {loyal:.1f}%"],
        ["Payment method", f"Electronic check: {echk:.1f}% ({n_echk:,} customers)",
         f"Credit card (automatic): {card:.1f}%"],
        ["Internet service", f"Fiber optic: {fiber:.1f}% ({n_fiber:,} customers)", f"DSL: {dsl:.1f}%"],
        ["Online security", f"Without: {sec_no:.1f}%", f"With: {sec_yes:.1f}%"],
        ["Tech support", f"Without: {ts_no:.1f}%", f"With: {ts_yes:.1f}%"],
        ["Senior citizen", f"Senior: {sen:.1f}%", f"Overall: {overall:.1f}%"],
        ["Paperless billing", f"Paperless: {pl:.1f}%", f"Overall: {overall:.1f}%"],
    ], columns=["Factor", "Higher-churn group", "Comparison group"])
    st.dataframe(findings, hide_index=True, width="stretch")
    imp = get_csv(config.FEATURE_IMPORTANCE_PATH).head(5)
    st.markdown("Most influential features for the selected model (permutation importance): "
                + ", ".join(f"`{f}`" for f in imp["feature"]) + ".")

    st.header("B. General recommendations (to be validated)")
    st.markdown(
        """
1. **Early-tenure onboarding.** Churn is concentrated in the first year. A structured onboarding
   check-in during the first months is a reasonable candidate intervention.
2. **Contract migration offers.** Month-to-month customers churn far more than contract customers.
   Offering an incentive to move to a 1- or 2-year plan is worth testing - but note that customers who
   *choose* long contracts may simply be more committed already (self-selection), so the
   effect of switching is unknown.
3. **Fiber-optic experience review.** Fiber customers churn at a high rate despite paying more.
   Investigate price perception, service quality and competitor offers for this segment.
4. **Bundle security / tech-support add-ons.** Customers with these add-ons churn less. Trial
   bundles could be tested, keeping in mind the association may reflect customer type rather
   than the add-on itself.
5. **Automatic payment nudges.** Electronic-check payers churn most. Encouraging auto-pay is
   low-cost to test; payment method may be a marker of engagement rather than a cause.
6. **Risk-ranked outreach.** Use the model's churn probability to prioritise a limited retention
   budget, and track the actual outcome of contacted vs. non-contacted customers.
"""
    )
    st.caption("No recommendation here has been experimentally validated on this dataset.")


# ----------------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------------
def main():
    st.sidebar.title("📉 Churn Analytics")
    page = st.sidebar.radio("Navigate", ["Project Overview", "Exploratory Data Analysis",
                                         "Model Evaluation", "Churn Prediction",
                                         "Retention Recommendations"])
    st.sidebar.markdown("---")
    st.sidebar.caption("Academic mini-project · CSE3141 Predictive Analytics · "
                       "Dataset: IBM Telco Customer Churn (sample data)")
    if not artifacts_ready():
        return
    df = get_data()
    summary = get_json(config.EDA_SUMMARY_PATH)
    meta = get_json(config.MODEL_METADATA_PATH)

    if page == "Project Overview":
        page_overview(df, summary, meta)
    elif page == "Exploratory Data Analysis":
        page_eda(df, summary)
    elif page == "Model Evaluation":
        page_models(meta)
    elif page == "Churn Prediction":
        page_predict(meta)
    else:
        page_retention(df, summary)


if __name__ == "__main__":
    main()

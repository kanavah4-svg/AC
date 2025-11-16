
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px

# Custom colour palette for charts
PALETTE = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692"]

@st.cache_data
def load_data():
    df = pd.read_csv("atelier8_survey_data.csv")
    return df

def main():
    st.set_page_config(
        page_title="ATELIER 8 – Circular Luxury Intelligence Dashboard",
        page_icon="👜",
        layout="wide"
    )

    # ---------- HEADER ----------
    st.markdown("<h1 style='margin-bottom:0px;'>ATELIER 8 – Circular Luxury Intelligence Dashboard</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#555; font-size:15px; margin-top:4px;'>"
        "Data-driven insights for restoration, authentication, and circular luxury in the UAE."
        "</p>",
        unsafe_allow_html=True
    )

    with st.expander("📌 What is this dashboard showing?", expanded=True):
        st.write(
            "- **Business idea**: ATELIER 8 is a circular luxury restoration & authentication studio for handbags and sneakers.\n"
            "- **Dataset**: Synthetic survey of 400 potential UAE luxury consumers (age, income, ownership, sustainability, willingness-to-pay).\n"
            "- **Goal**: Help a non-technical viewer understand **who the customers are**, **how to segment them**, and **how much they are willing to pay**."
        )

    df = load_data()

    # ---------- SIDEBAR FILTERS ----------
    st.sidebar.title("🎛 Filters")
    st.sidebar.write("Slice the dashboard for specific audiences.")

    age_filter = st.sidebar.multiselect(
        "Age group",
        options=sorted(df["age_group"].unique()),
        default=sorted(df["age_group"].unique())
    )
    income_filter = st.sidebar.multiselect(
        "Income level",
        options=sorted(df["income_level"].unique()),
        default=sorted(df["income_level"].unique())
    )
    gender_filter = st.sidebar.multiselect(
        "Gender",
        options=sorted(df["gender"].unique()),
        default=sorted(df["gender"].unique())
    )

    filtered_df = df[
        df["age_group"].isin(age_filter) &
        df["income_level"].isin(income_filter) &
        df["gender"].isin(gender_filter)
    ]

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Current sample after filters: **{len(filtered_df)} respondents**")

    tab_overview, tab_segments, tab_models, tab_upload = st.tabs(
        ["1️⃣ Overview & Story", "2️⃣ Customer Segments", "3️⃣ Predictive Models", "4️⃣ Upload & Score New Data"]
    )

    # ---------- TAB 1: OVERVIEW ----------
    with tab_overview:
        st.subheader("1️⃣ Business Overview – Who are our potential ATELIER 8 customers?")

        st.info(
            "This section answers: **Who did we survey? How ready are they to try ATELIER 8? "
            "How much are they willing to spend on restoration & authentication?**"
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Respondents (after filter)", len(filtered_df))
        with col2:
            early_adopters = (filtered_df["adoption_intent"] >= 4).mean() * 100
            st.metric("% Likely to Adopt (Adoption ≥ 4)", f"{early_adopters:.1f}%")
        with col3:
            avg_wtp_rest = filtered_df["wtp_restoration_aed"].mean()
            st.metric("Avg WTP – Restoration", f"AED {avg_wtp_rest:,.0f}")
        with col4:
            avg_wtp_auth = filtered_df["wtp_authentication_aed"].mean()
            st.metric("Avg WTP – Authentication", f"AED {avg_wtp_auth:,.0f}")

        # Extra high-level pies for non-technical viewers
        st.markdown("### A. Audience Mix at a Glance")
        pie_col1, pie_col2 = st.columns(2)
        with pie_col1:
            fig_pie_income = px.pie(
                filtered_df,
                names="income_level",
                title="Customer Split by Income Level",
                color="income_level",
                color_discrete_sequence=PALETTE
            )
            fig_pie_income.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_pie_income, use_container_width=True)
            st.caption("Shows which income bands dominate ATELIER 8's potential customer base.")

        with pie_col2:
            fig_pie_gender = px.pie(
                filtered_df,
                names="gender",
                title="Customer Split by Gender",
                color="gender",
                color_discrete_sequence=PALETTE
            )
            fig_pie_gender.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_pie_gender, use_container_width=True)
            st.caption("Helps decide whether to design communication more gender-neutral or slightly skewed.")

        st.markdown("### B. Adoption & Sustainability View")

        col_a, col_b = st.columns(2)
        with col_a:
            adoption_age = (
                filtered_df.groupby("age_group")["adoption_intent"]
                .mean()
                .reset_index()
                .sort_values("adoption_intent", ascending=False)
            )
            fig1 = px.bar(
                adoption_age,
                x="age_group",
                y="adoption_intent",
                color="age_group",
                color_discrete_sequence=PALETTE,
                title="Average Adoption Intention by Age Group",
                labels={"adoption_intent": "Adoption Intention (1–5)", "age_group": "Age Group"}
            )
            fig1.update_layout(showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
            st.caption("Younger and mid-30s luxury buyers tend to be slightly more open to trying ATELIER 8.")

        with col_b:
            fig2 = px.histogram(
                filtered_df,
                x="sustainability_importance",
                nbins=5,
                title="Sustainability Importance Distribution",
                labels={"sustainability_importance": "Sustainability Importance (1–5)"},
                color_discrete_sequence=[PALETTE[2]]
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("Most respondents rate sustainability from medium to high – a good fit for a circular brand story.")

        st.markdown("### C. Luxury Ownership & Willingness to Pay")

        col_c, col_d = st.columns(2)
        with col_c:
            brand_cols = ["owns_chanel", "owns_lv", "owns_dior", "owns_gucci", "owns_hermes", "owns_sneaker_grails"]
            brand_counts = (
                filtered_df[brand_cols].sum().reset_index()
                .rename(columns={"index": "brand", 0: "count"})
            )
            brand_counts.columns = ["brand", "count"]
            fig3 = px.bar(
                brand_counts,
                x="brand",
                y="count",
                title="Ownership of Key Luxury Brands",
                labels={"brand": "Brand", "count": "Number of Respondents"},
                color="brand",
                color_discrete_sequence=PALETTE
            )
            fig3.update_layout(showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)
            st.caption("LV, Gucci and sneaker grails dominate – ideal anchor brands for early ATELIER 8 campaigns.")

        with col_d:
            fig4 = px.box(
                filtered_df,
                x="income_level",
                y="wtp_restoration_aed",
                title="WTP for Restoration by Income Level",
                labels={"income_level": "Income Level", "wtp_restoration_aed": "WTP Restoration (AED)"},
                color="income_level",
                color_discrete_sequence=PALETTE
            )
            st.plotly_chart(fig4, use_container_width=True)
            st.caption("Higher income levels show both higher median WTP and wider spread – room for premium tiers.")

        st.markdown("### D. Sustainability ↔ WTP Relationship")

        fig5 = px.scatter(
            filtered_df,
            x="sustainability_importance",
            y="wtp_restoration_aed",
            color="income_level",
            title="Sustainability vs WTP for Restoration (coloured by Income Level)",
            labels={
                "sustainability_importance": "Sustainability Importance (1–5)",
                "wtp_restoration_aed": "WTP Restoration (AED)"
            },
            color_discrete_sequence=PALETTE
        )
        st.plotly_chart(fig5, use_container_width=True)
        st.caption(
            "Top-right dots represent ideal ATELIER 8 early adopters – high sustainability concern and high spend potential."
        )

    # ---------- TAB 2: SEGMENTS ----------
    with tab_segments:
        st.subheader("2️⃣ Customer Segments – Data-driven luxury personas")

        st.info(
            "We use K-Means clustering to group customers by **willingness to pay, ownership, sustainability and resale interest**.\n"
            "You can translate each cluster into a simple persona name for your report."
        )

        features = filtered_df[[
            "wtp_restoration_aed",
            "wtp_authentication_aed",
            "sustainability_importance",
            "handbags_owned",
            "sneakers_owned",
            "resale_interest"
        ]].copy()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)

        n_clusters = st.slider("Choose number of clusters (K)", min_value=2, max_value=6, value=4, step=1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        filtered_df["cluster"] = clusters

        col1_seg, col2_seg = st.columns([1.3, 1])
        with col1_seg:
            fig_seg = px.scatter(
                filtered_df,
                x="wtp_restoration_aed",
                y="wtp_authentication_aed",
                color="cluster",
                hover_data=["income_level", "age_group", "sustainability_importance", "resale_interest"],
                title="Customer Segments by WTP (Restoration vs Authentication)",
                labels={
                    "wtp_restoration_aed": "WTP Restoration (AED)",
                    "wtp_authentication_aed": "WTP Authentication (AED)",
                    "cluster": "Cluster"
                },
                color_discrete_sequence=PALETTE
            )
            st.plotly_chart(fig_seg, use_container_width=True)

        with col2_seg:
            cluster_summary = filtered_df.groupby("cluster")[
                ["wtp_restoration_aed", "wtp_authentication_aed", "sustainability_importance",
                 "handbags_owned", "sneakers_owned", "resale_interest"]
            ].mean().round(1)
            st.markdown("**Cluster Profile Summary (averages)**")
            st.dataframe(cluster_summary)

        st.markdown("#### How to talk about these clusters in class")
        st.write(
            "- Cluster with highest WTP and resale interest → **“Collectors / Rotation Enthusiasts”** – priority for concierge-level services.\n"
            "- Cluster with many sneakers and moderate WTP → **“Hype Sneaker Owners”** – focus on subscription cleaning.\n"
            "- Cluster with high sustainability but fewer items → **“Conscious Curators”** – targeted with trust & authenticity messaging."
        )

    # ---------- TAB 3: PREDICTIVE MODELS ----------
    with tab_models:
        st.subheader("3️⃣ Predictive Models – From insight to action")

        st.info(
            "These models show how ATELIER 8 can start using AI/ML in a simple, business-focused way – "
            "to score leads and design pricing."
        )

        model_tab1, model_tab2, model_tab3 = st.tabs(
            ["🧮 Classification: Who Adopts?", "💰 Regression: WTP for Restoration", "🧷 Association Patterns"]
        )

        # Classification
        with model_tab1:
            st.markdown("### 🧮 Classification – Likely vs unlikely adopters")

            df_clf = filtered_df.copy()
            df_clf["adopt_label"] = (df_clf["adoption_intent"] >= 4).astype(int)

            X = pd.get_dummies(df_clf[[
                "age_group", "income_level", "gender",
                "sustainability_importance", "owns_luxury_items",
                "handbags_owned", "sneakers_owned", "resale_interest"
            ]], drop_first=True)
            y = df_clf["adopt_label"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )

            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            st.write(f"**Accuracy on test data:** `{acc:.2f}` (1.00 = perfect, 0.50 = random guess)")

            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose().round(2)
            st.markdown("**Model performance by class**")
            st.dataframe(report_df)

            st.markdown("**How ATELIER 8 can use this:**")
            st.write(
                "- Score leads from Instagram, store walk-ins or website forms.\n"
                "- Prioritise top 30–40% probability for premium outreach and early trials."
            )

        # Regression
        with model_tab2:
            st.markdown("### 💰 Regression – Drivers of WTP for Restoration")

            df_reg = filtered_df.copy()
            X_reg = pd.get_dummies(df_reg[[
                "income_level",
                "sustainability_importance",
                "handbags_owned",
                "sneakers_owned",
                "resale_interest"
            ]], drop_first=True)
            y_reg = df_reg["wtp_restoration_aed"]

            reg = LinearRegression()
            reg.fit(X_reg, y_reg)

            coef_df = pd.DataFrame({
                "feature": X_reg.columns,
                "coefficient": reg.coef_
            }).sort_values("coefficient", ascending=False)
            st.write("**Model Coefficients – AED change for +1 unit in feature**")
            st.dataframe(coef_df)

            st.markdown("#### Scenario Simulator – Design an example customer")
            col_sim1, col_sim2, col_sim3 = st.columns(3)
            with col_sim1:
                sim_income = st.selectbox("Income level", ["Low", "Mid", "High", "Very High"])
                sim_sust = st.slider("Sustainability importance (1–5)", 1, 5, 4)
            with col_sim2:
                sim_bags = st.slider("Luxury handbags owned", 0, 10, 2)
                sim_sneakers = st.slider("Premium sneakers owned", 0, 10, 3)
            with col_sim3:
                sim_resale = st.slider("Resale interest (1–5)", 1, 5, 4)

            sim_row = pd.DataFrame({
                "income_level": [sim_income],
                "sustainability_importance": [sim_sust],
                "handbags_owned": [sim_bags],
                "sneakers_owned": [sim_sneakers],
                "resale_interest": [sim_resale]
            })
            sim_row_enc = pd.get_dummies(sim_row, columns=["income_level"], drop_first=True)
            sim_row_enc = sim_row_enc.reindex(columns=X_reg.columns, fill_value=0)

            pred_wtp = reg.predict(sim_row_enc)[0]
            st.success(f"Estimated fair WTP for restoration: **AED {pred_wtp:,.0f}**")

            st.markdown("**How ATELIER 8 can use this:**")
            st.write(
                "- Build **good–better–best** pricing tiers informed by data.\n"
                "- Avoid underpricing high-income collectors with high sustainability & resale interest."
            )

        # Association patterns
        with model_tab3:
            st.markdown("### 🧷 Association Patterns – Brand bundles & cross-sell ideas")

            brand_cols = ["owns_chanel", "owns_lv", "owns_dior", "owns_gucci", "owns_hermes", "owns_sneaker_grails"]
            brand_data = filtered_df[brand_cols]

            total_customers = len(brand_data)
            pairs = []
            for i, col_i in enumerate(brand_cols):
                for j, col_j in enumerate(brand_cols):
                    if i >= j:
                        continue
                    both = ((brand_data[col_i] == 1) & (brand_data[col_j] == 1)).sum()
                    support = both / total_customers
                    if support == 0:
                        continue
                    conf_i_j = both / (brand_data[col_i] == 1).sum() if (brand_data[col_i] == 1).sum() > 0 else 0
                    conf_j_i = both / (brand_data[col_j] == 1).sum() if (brand_data[col_j] == 1).sum() > 0 else 0
                    pairs.append({
                        "rule": f"{col_i} → {col_j}",
                        "support": round(support, 3),
                        "confidence": round(conf_i_j, 3)
                    })
                    pairs.append({
                        "rule": f"{col_j} → {col_i}",
                        "support": round(support, 3),
                        "confidence": round(conf_j_i, 3)
                    })

            assoc_df = pd.DataFrame(pairs).sort_values("confidence", ascending=False).head(10)
            st.write("**Top brand–brand association patterns**")
            st.dataframe(assoc_df)

            st.markdown("**Example business actions:**")
            st.write(
                "- If `owns_chanel → owns_lv` has high confidence, design **Chanel + LV care packages**.\n"
                "- Offer follow-up deals: "You restored your Chanel flap, add LV Neverfull spa at 20% off.""
            )

    # ---------- TAB 4: UPLOAD & SCORE ----------
    with tab_upload:
        st.subheader("4️⃣ Upload & Score New Leads")

        st.info(
            "Once ATELIER 8 has real customer data, this section can score each lead with an **adoption probability**."
        )

        st.markdown("**Your CSV should include at least these columns:**")
        st.code(
            "age_group, income_level, gender, sustainability_importance, owns_luxury_items, "
            "handbags_owned, sneakers_owned, resale_interest",
            language="text"
        )

        uploaded = st.file_uploader("Upload CSV file", type=["csv"])

        if uploaded is not None:
            user_df = pd.read_csv(uploaded)
            st.write("Preview of uploaded data:")
            st.dataframe(user_df.head())

            try:
                df_clf = df.copy()
                df_clf["adopt_label"] = (df_clf["adoption_intent"] >= 4).astype(int)
                X_base = pd.get_dummies(df_clf[[
                    "age_group", "income_level", "gender",
                    "sustainability_importance", "owns_luxury_items",
                    "handbags_owned", "sneakers_owned", "resale_interest"
                ]], drop_first=True)
                y_base = df_clf["adopt_label"]

                clf_full = LogisticRegression(max_iter=1000)
                clf_full.fit(X_base, y_base)

                X_user = pd.get_dummies(user_df[[
                    "age_group", "income_level", "gender",
                    "sustainability_importance", "owns_luxury_items",
                    "handbags_owned", "sneakers_owned", "resale_interest"
                ]], drop_first=True)

                X_user = X_user.reindex(columns=X_base.columns, fill_value=0)
                probs = clf_full.predict_proba(X_user)[:, 1]

                user_df["adoption_probability"] = np.round(probs, 3)
                st.success("Scored adoption probabilities for uploaded leads.")
                st.dataframe(user_df.head())

                csv = user_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Scored CSV",
                    data=csv,
                    file_name="atelier8_scored_leads.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error while scoring data: {e}")

    st.markdown("---")
    st.caption("ATELIER 8 · Circular Luxury · Data-Driven Stewardship")

if __name__ == "__main__":
    main()

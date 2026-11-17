import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Mall Customer Segments", page_icon="🛍️")


# 1. Load the saved model, scaler, segment names and the customer data (only once)
@st.cache_resource
def load_everything():
    saved = joblib.load("segment_model.pkl")
    customers = pd.read_csv("Mall_Customers_clean.csv")
    return saved, customers

saved, customers = load_everything()
model, scaler, names, tips = saved["model"], saved["scaler"], saved["names"], saved["tips"]

# 2. Page title
st.title("🛍️ Which customer group is this?")
st.write("Move the sliders to describe a customer. The K-means model puts them in one of **5 groups**.")

# 3. Inputs
income = st.slider("Annual income (thousand $)", 10, 150, 60)
spending = st.slider("Spending score (1 = spends very little, 100 = spends a lot)", 1, 100, 50)

# 4. Predict the group: scale the numbers the same way as in training, then ask the model
new_customer = pd.DataFrame({"income": [income], "spending": [spending]})
group = int(model.predict(scaler.transform(new_customer))[0])

st.subheader(f"Group {group}: {names[group]}")
st.info(f"💡 Marketing idea: {tips[group]}")

# 5. Chart: all customers coloured by group, plus the new customer as a big star
all_groups = model.predict(scaler.transform(customers[["income", "spending"]]))
fig, ax = plt.subplots(figsize=(7, 5))
scatter = ax.scatter(customers["income"], customers["spending"], c=all_groups, cmap="tab10", vmin=0, vmax=9, s=40, alpha=0.7)
ax.scatter(income, spending, marker="*", s=600, c="black", edgecolors="yellow", linewidths=1.5, label="This customer")
ax.set_xlabel("Annual income (thousand $)")
ax.set_ylabel("Spending score")
ax.legend(loc="upper center")
st.pyplot(fig)

# 6. Short table of the groups
st.write("**The 5 groups**")
st.table(pd.DataFrame({"Group": list(names.keys()), "Name": list(names.values())}).set_index("Group"))

from utils import predict_news

# Example news to test
news_text = """
At least 118 Palestinians, including 12 aid seekers, have been killed and 581 others injured in Israeli attacks across the Gaza Strip in the past 24 hours, according to the enclave’s Health Ministry.
Francesca Albanese, the UN special rapporteur on human rights in the occupied Palestinian territory, has called on countries to impose a full arms embargo and cut off trade and financial ties with Israel, which she accused of waging a “genocidal campaign” in Gaza.
Meanwhile, Hamas says it is studying a new proposal for a temporary ceasefire in Gaza, after US President Donald Trump said Israel had agreed to a 60-day truce.
Israel’s war on Gaza has killed at least 57,130 people and wounded 134,592, according to Gaza’s Health Ministry. An estimated 1,139 people were killed in Israel during the October 7, 2023 attacks, and more than 200 were taken captive.
"""

# Get prediction
result = predict_news(news_text)
print("Prediction:", result)

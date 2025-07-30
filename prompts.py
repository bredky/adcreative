def generate_pros_cons_prompt(predicted_ctr):
    return f"""
You are an expert advertising strategist.

A client has submitted a new static ad creative. The model predicts a click-through rate (CTR) of {predicted_ctr:.4f} for this ad.

Analyze this ad based on the following industry-proven criteria:
- Visual appeal and design clarity
- Message clarity and strength of call-to-action
- Storytelling or outcome demonstration
- Use of trust-building elements (testimonials, press)
- Brand visibility and identity
- Relevance to audience or timing
- Alignment with performance KPIs (CTR, conversions)

Return 2 bullet points each for:
- Pros: What this ad likely does well
- Cons: What this ad could improve
"""

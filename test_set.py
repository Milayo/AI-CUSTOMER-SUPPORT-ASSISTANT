"""
Fields:
  question            - what a real customer would type
  should_answer       - True if the docs cover this; False if it must decline/escalate
  context_must_contain- a distinctive string that MUST appear in the retrieved
                        chunks for the answer to be possible (retrieval check)
  expected_answer     - the reference answer, used by the LLM judge
"""

TEST_SET = [
    
    {
        "question": "where's my package?",
        "should_answer": True,
        "context_must_contain": "tracking number",
        "expected_answer": "A shipping confirmation email with a tracking number is sent once the order is dispatched. Tracking can take up to 48 hours to activate. Customers can also check Order History in their account.",
    },
    {
        "question": "can I send back a swimsuit I bought?",
        "should_answer": True,
        "context_must_contain": "swimwear",
        "expected_answer": "No. Swimwear is final sale and cannot be returned or exchanged, for hygiene reasons.",
    },
    {
        "question": "how long until I get my money back?",
        "should_answer": True,
        "context_must_contain": "5 to 10 business days",
        "expected_answer": "Approved refunds are processed within 5 to 10 business days to the original payment method, and the bank may take a further 5 to 10 business days to show it.",
    },
    {
        "question": "do you take PayPal?",
        "should_answer": True,
        "context_must_contain": "PayPal",
        "expected_answer": "Yes. Acme Store accepts Visa, Mastercard, American Express, PayPal, Apple Pay, and Google Pay.",
    },
    {
        "question": "how long will delivery to Nigeria take?",
        "should_answer": True,
        "context_must_contain": "Nigeria",
        "expected_answer": "Delivery to Nigeria takes 15 to 25 business days.",
    },
    {
        "question": "I typed the wrong address, can it be fixed?",
        "should_answer": True,
        "context_must_contain": "dispatched",
        "expected_answer": "The shipping address can only be changed before the order is dispatched. Contact support immediately with the order number and corrected address.",
    },
    {
        "question": "is delivery free?",
        "should_answer": True,
        "context_must_contain": "free standard shipping",
        "expected_answer": "Standard shipping is a flat 6.99 USD for orders under 50 USD, and free for orders of 50 USD or more.",
    },
    {
        "question": "can I stack two promo codes?",
        "should_answer": True,
        "context_must_contain": "one discount code",
        "expected_answer": "No. Only one discount code can be applied per order, and codes cannot be combined with clearance pricing.",
    },
    {
        "question": "can you deliver to a PO box?",
        "should_answer": True,
        "context_must_contain": "PO Box",
        "expected_answer": "No. Acme Store does not deliver to PO Boxes; a physical street address is required.",
    },
    {
        "question": "my order turned up smashed, what now?",
        "should_answer": True,
        "context_must_contain": "damaged",
        "expected_answer": "Contact support within 7 days of delivery with the order number and photos of the item and packaging. A replacement or full refund including shipping will be arranged at no cost.",
    },
    {
        "question": "I forgot my login details",
        "should_answer": True,
        "context_must_contain": "Forgot Password",
        "expected_answer": "Select Forgot Password on the login page and enter the registered email. A reset link is emailed and is valid for 24 hours.",
    },
    {
        "question": "who covers the cost of posting an item back?",
        "should_answer": True,
        "context_must_contain": "return shipping",
        "expected_answer": "Acme Store pays return shipping if the item was defective, damaged, or incorrect. Otherwise the customer pays return shipping.",
    },

    # ---------- NOT answerable: must decline / escalate ----------
    {
        "question": "who is your CEO?",
        "should_answer": False,
        "context_must_contain": None,
        "expected_answer": "Should decline: this information is not in the help documents.",
    },
    {
        "question": "do you sell office furniture?",
        "should_answer": False,
        "context_must_contain": None,
        "expected_answer": "Should decline: product catalogue information is not in the help documents.",
    },
    {
        "question": "how long does shipping to Antarctica take?",
        "should_answer": False,
        "context_must_contain": None,
        "expected_answer": "Should decline: Antarctica is not listed in the delivery destinations.",
    },
    {
        "question": "are you hiring right now?",
        "should_answer": False,
        "context_must_contain": None,
        "expected_answer": "Should decline: recruitment information is not in the help documents.",
    },
]
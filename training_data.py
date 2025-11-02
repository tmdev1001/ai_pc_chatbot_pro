"""
Training data for the chatbot intent classification model.
Contains the embedding function and all training examples.
"""

import random

# Fake embedding (in real case use BERT or SentenceTransformer)
def simple_embed(text):
    """Convert text to a simple feature vector."""
    return [len(text), text.count('?'), text.count('')]  # trivial feature


def generate_additional_samples(count=10000):
    """Generate additional diverse training samples programmatically."""
    samples = []
    random.seed(42)  # For reproducibility
    
    # Intent 0 patterns (most_used)
    starters_0 = ["What", "Which", "Tell me", "Show", "Display", "Give me", "List", "Find"]
    app_words = ["app", "application", "program", "software", "tool", "application"]
    verbs_0 = ["I used", "I have used", "I've used", "I'm using", "I use", "I opened", "I launched", "did I use"]
    qualifiers_0 = ["most", "the most", "top", "highest", "primary", "main", "number one", "best", "favorite"]
    time_refs = ["", " today", " this week", " recently", " this month", " lately"]
    
    # Intent 1 patterns (usage_summary)
    starters_1 = ["How much", "What's my", "Show me", "Tell me", "Display", "Give me", "What is my"]
    time_words = ["time", "usage", "activity", "duration"]
    verbs_1 = ["did I spend", "have I spent", "I spent", "I used", "was spent", "total", "overall"]
    qualifiers_1 = ["total", "overall", "cumulative", "combined", "all", "entire"]
    stats_words = ["statistics", "summary", "report", "data", "analytics", "breakdown", "overview"]
    
    # Generate Intent 0 samples (5000)
    for i in range(count // 2):
        pattern_type = random.random()
        if pattern_type < 0.3:  # "What/Which app I used most?"
            text = f"{random.choice(starters_0)} {random.choice(app_words)} {random.choice(verbs_0)} {random.choice(qualifiers_0)}{random.choice(time_refs)}"
        elif pattern_type < 0.6:  # "Show me the most used app"
            text = f"{random.choice(['Show', 'Display', 'Tell me', 'Give me'])} {random.choice(['the', 'my', ''])} {random.choice(qualifiers_0)} {random.choice(['used', 'opened'])} {random.choice(app_words)}"
        else:  # "Which application tops my usage?"
            text = f"{random.choice(starters_0)} {random.choice(app_words)} {random.choice(['tops', 'leads', 'ranks first in', 'dominates'])} {random.choice(['my', 'the'])} usage{random.choice(time_refs)}"
        
        # Add question mark or not
        if random.random() > 0.25:  # 75% have question marks
            text = text.rstrip("?") + "?"
        text = " ".join(text.split())  # Clean up whitespace
        samples.append((simple_embed(text), 0))
    
    # Generate Intent 1 samples (5000)
    for i in range(count // 2):
        pattern_type = random.random()
        if pattern_type < 0.35:  # "How much time did I spend?"
            text = f"How much {random.choice(time_words)} {random.choice(['did I', 'have I', ''])} {random.choice(['spend', 'use', 'spent', 'used'])}?"
        elif pattern_type < 0.65:  # "What's my total usage?"
            text = f"{random.choice(['What\'s my', 'What is my', 'Show my'])} {random.choice(qualifiers_1)} {random.choice(time_words)}?"
        elif pattern_type < 0.85:  # "Show me usage statistics"
            text = f"{random.choice(['Show', 'Display', 'Tell me', 'Give me'])} {random.choice(['usage', 'time', 'activity'])} {random.choice(stats_words)}"
            if random.random() > 0.3:
                text += "?"
        else:  # "Tell me my total usage time"
            text = f"{random.choice(['Tell me', 'Show me', 'Give me'])} {random.choice(['my', 'the', ''])} {random.choice(qualifiers_1)} {random.choice(['usage', 'time', 'activity'])} {random.choice(['time', '', 'duration'])}"
            if random.random() > 0.3:
                text = text.rstrip("?") + "?"
        
        text = " ".join(text.split())  # Clean up whitespace
        samples.append((simple_embed(text), 1))
    
    random.seed()  # Reset seed
    return samples


def get_training_data():
    """
    Returns the training dataset with embedded text and intent labels.
    
    Intent labels:
    0 = most_used (questions about which app was used most)
    1 = usage_summary (questions about usage time/statistics)
    """
    data = [
        (simple_embed("Which app I used most?"), 0),  # 0=most_used
        (simple_embed("show usage time?"), 1),  # 1=usage_summary
        # Intent 0: Most Used App variations (51 examples)
        (simple_embed("What's my most used app?"), 0),
        (simple_embed("Which application did I use the most?"), 0),
        (simple_embed("Show me the app I used most"), 0),
        (simple_embed("What app have I been using the most?"), 0),
        (simple_embed("Which program is my top used app?"), 0),
        (simple_embed("What's the most frequently used app?"), 0),
        (simple_embed("Tell me my most used application"), 0),
        (simple_embed("Which app tops my usage list?"), 0),
        (simple_embed("What application have I opened most?"), 0),
        (simple_embed("Show the app with highest usage"), 0),
        (simple_embed("Which app do I use most often?"), 0),
        (simple_embed("What's my top app?"), 0),
        (simple_embed("Which software did I use most?"), 0),
        (simple_embed("What app is used the most by me?"), 0),
        (simple_embed("Display my most used app"), 0),
        (simple_embed("Which application is most popular on my PC?"), 0),
        (simple_embed("What's the app I launch most?"), 0),
        (simple_embed("Show top app by usage"), 0),
        (simple_embed("Which app has the most usage time?"), 0),
        (simple_embed("What application appears most in my usage?"), 0),
        (simple_embed("Tell me which app I use most"), 0),
        (simple_embed("What's the number one app I use?"), 0),
        (simple_embed("Which program do I open most?"), 0),
        (simple_embed("Show me the top used application"), 0),
        (simple_embed("What app ranks first in my usage?"), 0),
        (simple_embed("Which app is my favorite based on usage?"), 0),
        (simple_embed("What application do I use most frequently?"), 0),
        (simple_embed("Show my most used software"), 0),
        (simple_embed("Which app have I been spending most time on?"), 0),
        (simple_embed("What's my primary app?"), 0),
        (simple_embed("Which application tops the chart?"), 0),
        (simple_embed("What app did I use most today?"), 0),
        (simple_embed("Show the most opened app"), 0),
        (simple_embed("Which program is my go-to app?"), 0),
        (simple_embed("What application leads in usage?"), 0),
        (simple_embed("Tell me my top app"), 0),
        (simple_embed("Which app comes first in usage?"), 0),
        (simple_embed("What's the app with maximum usage?"), 0),
        (simple_embed("Show app that I use most"), 0),
        (simple_embed("Which application is number one?"), 0),
        (simple_embed("What app dominates my usage?"), 0),
        (simple_embed("Display the most used application"), 0),
        (simple_embed("Which program have I used most?"), 0),
        (simple_embed("What's my highest usage app?"), 0),
        (simple_embed("Show me top application"), 0),
        (simple_embed("show app mosed used "), 0),
        (simple_embed("Which app is used most frequently?"), 0),
        (simple_embed("What application has the highest usage?"), 0),
        (simple_embed("Tell me the app I use most"), 0),
        (simple_embed("Which software tops my usage?"), 0),
        (simple_embed("What app do I access most?"), 0),
        (simple_embed("Show the application I use most"), 0),
        # Intent 1: Usage Summary variations (50 examples)
        (simple_embed("How much time did I spend?"), 1),
        (simple_embed("What's my total usage time?"), 1),
        (simple_embed("Show me usage statistics"), 1),
        (simple_embed("Display my usage summary"), 1),
        (simple_embed("What's the total time I used apps?"), 1),
        (simple_embed("How long have I been using apps?"), 1),
        (simple_embed("Show total active time"), 1),
        (simple_embed("What's my overall usage?"), 1),
        (simple_embed("Give me a usage report"), 1),
        (simple_embed("How many minutes did I use apps?"), 1),
        (simple_embed("Show usage statistics"), 1),
        (simple_embed("What's the summary of my usage?"), 1),
        (simple_embed("Display total time spent"), 1),
        (simple_embed("How much time total?"), 1),
        (simple_embed("Show me my usage data"), 1),
        (simple_embed("What's my usage overview?"), 1),
        (simple_embed("Tell me my total usage time"), 1),
        (simple_embed("How long have I been active?"), 1),
        (simple_embed("Show total app usage time"), 1),
        (simple_embed("What's the cumulative usage?"), 1),
        (simple_embed("Display usage breakdown"), 1),
        (simple_embed("How much time in total?"), 1),
        (simple_embed("Show usage analytics"), 1),
        (simple_embed("What's my time spent summary?"), 1),
        (simple_embed("Give me usage details"), 1),
        (simple_embed("How many hours did I use?"), 1),
        (simple_embed("Show total time"), 1),
        (simple_embed("What's my overall activity time?"), 1),
        (simple_embed("Display usage information"), 1),
        (simple_embed("How much total usage?"), 1),
        (simple_embed("Show me usage totals"), 1),
        (simple_embed("What's the total active duration?"), 1),
        (simple_embed("Tell me total usage"), 1),
        (simple_embed("How long total time?"), 1),
        (simple_embed("Show usage summary"), 1),
        (simple_embed("What's my cumulative time?"), 1),
        (simple_embed("Display total activity"), 1),
        (simple_embed("How much time altogether?"), 1),
        (simple_embed("Show overall usage"), 1),
        (simple_embed("What's the total duration?"), 1),
        (simple_embed("Give me total usage stats"), 1),
        (simple_embed("How many minutes total?"), 1),
        (simple_embed("Show usage report"), 1),
        (simple_embed("What's my total activity?"), 1),
        (simple_embed("Display time summary"), 1),
        (simple_embed("How much usage time total?"), 1),
        (simple_embed("Show me the totals"), 1),
        (simple_embed("What's the usage breakdown?"), 1),
        (simple_embed("Tell me usage time summary"), 1),
        (simple_embed("How long total?"), 1),
        (simple_embed("Show total usage statistics"), 1),
    ]
    
    # Generate 10,000 additional diverse samples
    additional_samples = generate_additional_samples(10000)
    data.extend(additional_samples)
    
    return data


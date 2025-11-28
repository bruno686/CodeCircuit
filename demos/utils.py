import html

import torch
from IPython.display import HTML, display


def get_topk(logits: torch.Tensor, tokenizer, k: int = 5):
    probs = torch.softmax(logits.squeeze()[-1], dim=-1)
    topk = torch.topk(probs, k)
    return [(tokenizer.decode([topk.indices[i]]), topk.values[i].item()) for i in range(k)]


# Now let's create a version that's more adaptive to dark/light mode
def display_topk_token_predictions(sentence, original_logits, new_logits, tokenizer, k: int = 5):
    """
    Version that tries to be more adaptive to both dark and light modes
    using higher contrast elements and CSS variables where possible
    """

    original_tokens = get_topk(original_logits, tokenizer, k)
    new_tokens = get_topk(new_logits, tokenizer, k)

    # This version uses a technique that will work better in dark mode
    # by using a combination of background colors and border styling
    html = f"""
    <style>
    .token-viz {{
        font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
        margin-bottom: 10px;
        max-width: 700px;
    }}
    .token-viz .header {{
        font-weight: bold;
        font-size: 14px;
        margin-bottom: 3px;
        padding: 4px 6px;
        border-radius: 3px;
        color: white;
        display: inline-block;
    }}
    .token-viz .sentence {{
        background-color: rgba(200, 200, 200, 0.2);
        padding: 4px 6px;
        border-radius: 3px;
        border: 1px solid rgba(100, 100, 100, 0.5);
        font-family: monospace;
        margin-bottom: 8px;
        font-weight: 500;
        font-size: 14px;
    }}
    .token-viz table {{
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 8px;
        font-size: 13px;
        table-layout: fixed;
    }}
    .token-viz th {{
        text-align: left;
        padding: 4px 6px;
        font-weight: bold;
        border: 1px solid rgba(150, 150, 150, 0.5);
        background-color: rgba(200, 200, 200, 0.3);
    }}
    .token-viz td {{
        padding: 3px 6px;
        border: 1px solid rgba(150, 150, 150, 0.5);
        font-weight: 500;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }}
    .token-viz .token-col {{
        width: 20%;
    }}
    .token-viz .prob-col {{
        width: 15%;
    }}
    .token-viz .dist-col {{
        width: 65%;
    }}
    .token-viz .monospace {{
        font-family: monospace;
    }}
    .token-viz .bar-container {{
        display: flex;
        align-items: center;
    }}
    .token-viz .bar {{
        height: 12px;
        min-width: 2px;
    }}
    .token-viz .bar-text {{
        margin-left: 6px;
        font-weight: 500;
        font-size: 12px;
    }}
    .token-viz .even-row {{
        background-color: rgba(240, 240, 240, 0.1);
    }}
    .token-viz .odd-row {{
        background-color: rgba(255, 255, 255, 0.1);
    }}
    </style>
    
    <div class="token-viz">
        <div class="header" style="background-color: #555555;">Input Sentence:</div>
        <div class="sentence">{sentence}</div>
        
        <div>
            <div class="header" style="background-color: #2471A3;">Original Top {k} Tokens</div>
            <table>
                <thead>
                    <tr>
                        <th class="token-col">Token</th>
                        <th class="prob-col" style="text-align: right;">Probability</th>
                        <th class="dist-col">Distribution</th>
                    </tr>
                </thead>
                <tbody>
    """

    # Calculate max probability for scaling
    max_prob = max(
        max([prob for _, prob in original_tokens]), max([prob for _, prob in new_tokens])
    )

    # Add rows for original tokens
    for i, (token, prob) in enumerate(original_tokens):
        bar_width = int(prob / max_prob * 100)
        row_class = "even-row" if i % 2 == 0 else "odd-row"
        html += f"""
                    <tr class="{row_class}">
                        <td class="monospace token-col" title="{token}">{token}</td>
                        <td class="prob-col" style="text-align: right;">{prob:.3f}</td>
                        <td class="dist-col">
                            <div class="bar-container">
                                <div class="bar" style="background-color: #2471A3; width: {bar_width}%;"></div>
                                <span class="bar-text">{prob * 100:.1f}%</span>
                            </div>
                        </td>
                    </tr>
        """

    # Add new tokens table
    html += f"""
                </tbody>
            </table>
            
            <div class="header" style="background-color: #27AE60;">New Top {k} Tokens</div>
            <table>
                <thead>
                    <tr>
                        <th class="token-col">Token</th>
                        <th class="prob-col" style="text-align: right;">Probability</th>
                        <th class="dist-col">Distribution</th>
                    </tr>
                </thead>
                <tbody>
    """

    # Add rows for new tokens
    for i, (token, prob) in enumerate(new_tokens):
        bar_width = int(prob / max_prob * 100)
        row_class = "even-row" if i % 2 == 0 else "odd-row"
        html += f"""
                    <tr class="{row_class}">
                        <td class="monospace token-col" title="{token}">{token}</td>
                        <td class="prob-col" style="text-align: right;">{prob:.3f}</td>
                        <td class="dist-col">
                            <div class="bar-container">
                                <div class="bar" style="background-color: #27AE60; width: {bar_width}%;"></div>
                                <span class="bar-text">{prob * 100:.1f}%</span>
                            </div>
                        </td>
                    </tr>
        """

    html += """
                </tbody>
            </table>
        </div>
    </div>
    """

    display(HTML(html))


def display_generations_comparison(original_text, pre_intervention_gens, post_intervention_gens):
    """
    Display a comparison of pre-intervention and post-intervention generations
    with the new/continuation text highlighted.
    """
    # Ensure the original text is properly escaped
    escaped_original = html.escape(original_text)

    # Build the HTML with CSS for styling
    html_content = """
    <style>
    .generations-viz {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
        margin-bottom: 12px;
        font-size: 13px;
        max-width: 700px;
    }
    .generations-viz .section-header {
        font-weight: bold;
        font-size: 14px;
        margin: 10px 0 5px 0;
        padding: 4px 6px;
        border-radius: 3px;
        color: white;
        display: block;
    }
    .generations-viz .pre-intervention-header {
        background-color: #2471A3;
    }
    .generations-viz .post-intervention-header {
        background-color: #27AE60;
    }
    .generations-viz .generation-container {
        margin-bottom: 8px;
        padding: 3px;
        border-left: 3px solid rgba(100, 100, 100, 0.5);
    }
    .generations-viz .generation-text {
        background-color: rgba(200, 200, 200, 0.2);
        padding: 6px 8px;
        border-radius: 3px;
        border: 1px solid rgba(100, 100, 100, 0.5);
        font-family: monospace;
        font-weight: 500;
        white-space: pre-wrap;
        line-height: 1.2;
        font-size: 13px;
        overflow-x: auto;
    }
    .generations-viz .base-text {
        color: rgba(100, 100, 100, 0.9);
    }
    .generations-viz .new-text {
        background-color: rgba(255, 255, 0, 0.25);
        font-weight: bold;
        padding: 1px 0;
        border-radius: 2px;
    }
    .generations-viz .pre-intervention-item {
        border-left-color: #2471A3;
    }
    .generations-viz .post-intervention-item {
        border-left-color: #27AE60;
    }
    .generations-viz .generation-number {
        font-weight: bold;
        margin-bottom: 3px;
        color: rgba(70, 70, 70, 0.9);
        font-size: 12px;
    }
    </style>
    
    <div class="generations-viz">
    """

    # Add pre-intervention section
    html_content += """
    <div class="section-header pre-intervention-header">Pre-intervention generations:</div>
    """

    # Add each pre-intervention generation
    for i, gen_text in enumerate(pre_intervention_gens):
        # Split the text to highlight the continuation
        if gen_text.startswith(original_text):
            base_part = html.escape(original_text)
            new_part = html.escape(gen_text[len(original_text) :])
            formatted_text = f'<span class="base-text">{base_part}</span><span class="new-text">{new_part}</span>'
        else:
            formatted_text = html.escape(gen_text)

        html_content += f"""
        <div class="generation-container pre-intervention-item">
            <div class="generation-number">Generation {i + 1}</div>
            <div class="generation-text">{formatted_text}</div>
        </div>
        """

    # Add post-intervention section
    html_content += """
    <div class="section-header post-intervention-header">Post-intervention generations:</div>
    """

    # Add each post-intervention generation
    for i, gen_text in enumerate(post_intervention_gens):
        # Split the text to highlight the continuation
        if gen_text.startswith(original_text):
            base_part = html.escape(original_text)
            new_part = html.escape(gen_text[len(original_text) :])
            formatted_text = f'<span class="base-text">{base_part}</span><span class="new-text">{new_part}</span>'
        else:
            formatted_text = html.escape(gen_text)

        html_content += f"""
        <div class="generation-container post-intervention-item">
            <div class="generation-number">Generation {i + 1}</div>
            <div class="generation-text">{formatted_text}</div>
        </div>
        """

    html_content += """
    </div>
    """

    display(HTML(html_content))

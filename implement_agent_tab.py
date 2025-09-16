#!/usr/bin/env python3
"""Script to implement the new Agent tab structure and clean up temporary files"""

import re
import os

# Read the new agent tab HTML
with open('/Users/ishraq21/ragnetic/agent_tab_structure.html', 'r') as f:
    new_agent_html = f.read()

# Read the new agent tab CSS
with open('/Users/ishraq21/ragnetic/agent_tab_styles.css', 'r') as f:
    new_agent_css = f.read()

# Read the new agent tab JavaScript
with open('/Users/ishraq21/ragnetic/agent_tab_script.js', 'r') as f:
    new_agent_js = f.read()

# Read the current dashboard.html
with open('/Users/ishraq21/ragnetic/templates/dashboard.html', 'r') as f:
    html_content = f.read()

# Read the current dashboard.css
with open('/Users/ishraq21/ragnetic/static/css/dashboard.css', 'r') as f:
    css_content = f.read()

# 1. Replace the agents view content
# Find the start of the agents view
agents_start = html_content.find('<div class="view-content" id="agents-view">')
if agents_start != -1:
    # Find the end of the agents view (next view-content div)
    agents_end = html_content.find('<div class="view-content"', agents_start + 1)
    if agents_end == -1:
        # If no next view-content, find the closing div
        agents_end = html_content.find('</div>', agents_start + 1)
        # Find the matching closing div
        div_count = 1
        pos = agents_start + 1
        while div_count > 0 and pos < len(html_content):
            if html_content[pos:pos+6] == '<div c':
                div_count += 1
            elif html_content[pos:pos+6] == '</div>':
                div_count -= 1
            pos += 1
        if div_count == 0:
            agents_end = pos
    
    # Replace the entire agents section
    html_content = html_content[:agents_start] + new_agent_html + html_content[agents_end:]

# 2. Add the agent tab CSS
css_content += '\n\n' + new_agent_css

# 3. Add the agent tab JavaScript before the closing head tag
html_content = html_content.replace('</head>', f'    <script>{new_agent_js}\n    </script>\n</head>')

# Write the updated files
with open('/Users/ishraq21/ragnetic/templates/dashboard.html', 'w') as f:
    f.write(html_content)

with open('/Users/ishraq21/ragnetic/static/css/dashboard.css', 'w') as f:
    f.write(css_content)

# 4. Clean up temporary files
temp_files = [
    'agent_tab_structure.html',
    'agent_tab_styles.css', 
    'agent_tab_script.js',
    'implement_agent_tab.py'
]

for file in temp_files:
    if os.path.exists(file):
        os.remove(file)
        print(f"Deleted temporary file: {file}")

print("✅ Professional Agent Tab Implemented!")
print("")
print("🎯 Features Added:")
print("• Professional header with 'New Agent' CTA")
print("• Advanced search and filtering (status, model, deployment)")
print("• Table/Grid view toggle")
print("• Comprehensive agent table with all required fields:")
print("  - Agent Name (clickable to detail page)")
print("  - Status (Online/Offline/Error with colored pills)")
print("  - Model & Embedding Model")
print("  - Deployment Type (Not deployed/Chat/API/Slack/Discord)")
print("  - Last Run timestamp")
print("  - Cost This Month")
print("  - Actions (Deploy/Edit/Delete/Clone/Test)")
print("• Beautiful card view for visual users")
print("• Responsive design for all screen sizes")
print("• Professional styling matching AWS console standards")
print("")
print("🚀 Ready for agent management workflow!")

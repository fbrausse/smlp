#!/usr/bin/env python3.11
import requests
from bs4 import BeautifulSoup
import re
from sys import argv
from os.path import basename

def get_commit_info_from_github_release(url):
    """
    Extracts the commit hash and commit date from a GitHub release page URL
    without using git.

    Args:
        url (str): The URL of the GitHub release page (e.g.,
                   "https://github.com/SMLP-Systems/smlp/releases/tag/v1.2.3rc7").

    Returns:
        dict or None: A dictionary with 'commit_hash' and 'commit_date' if found,
                      otherwise None.
    """
    headers = {
        # Mimic a real browser to avoid being blocked by GitHub
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL '{url}': {e}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')

    commit_hash = None
    commit_date = None

    commit_link_pattern = re.compile(r'/commit/([0-9a-fA-F]{40})')

    # --- Strategy 1: Find the commit link and look for a sibling <time> element ---
    for link in soup.find_all('a', href=True):
        match = commit_link_pattern.search(link['href'])
        if match:
            commit_hash = match.group(1)

            # Walk up the DOM tree to find a nearby <time> or <relative-time> tag
            # GitHub places the commit timestamp close to the commit link
            parent = link.parent
            for _ in range(6):  # Search up to 6 levels up the DOM
                if parent is None:
                    break

                # Look for <relative-time> (GitHub's custom element) or <time>
                time_tag = parent.find(['relative-time', 'time'])
                if time_tag:
                    # 'datetime' attribute holds the ISO 8601 timestamp
                    commit_date = time_tag.get('datetime') or time_tag.get_text(strip=True)
                    break

                parent = parent.parent

            # Stop after the first valid commit link is found
            break

    # --- Strategy 2: Fallback - scan ALL <time> / <relative-time> tags on page ---
    if commit_hash and not commit_date:
        for time_tag in soup.find_all(['relative-time', 'time']):
            dt = time_tag.get('datetime')
            if dt:
                commit_date = dt
                break

    if not commit_hash:
        print("Could not find a commit hash on the page.")
        return None

    return {
        "commit_hash": commit_hash,
        "commit_date": commit_date or "Date not found"
    }


# --- Example Usage ---
if __name__ == "__main__":
    if len(argv) < 2:
        print(f"\nUsage: {basename(argv[0])} <tag name>\n")
        exit(1)

    release_url = f"https://github.com/SMLP-Systems/smlp/releases/tag/{argv[1]}"

    print(f"Fetching release info from:\n  {release_url}\n")
    info = get_commit_info_from_github_release(release_url)

    if info:
        print(f"  Commit Hash : {info['commit_hash']}")
        print(f"  Commit Date : {info['commit_date']}")
    else:
        print("Failed to retrieve release information.")


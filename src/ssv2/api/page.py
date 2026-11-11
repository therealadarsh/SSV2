"""The search page served at ``/``.

Kept as a Python string so the installed package (and the container image) always
contains it. No build step, no external assets: the page works offline.
"""

SEARCH_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SSV2 search</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 40rem;
    margin: 2rem auto; padding: 0 1rem; line-height: 1.4; }
  form { display: grid; gap: 0.5rem; }
  label { font-weight: 600; }
  input { font: inherit; padding: 0.4rem; width: 100%; box-sizing: border-box; }
  button { font: inherit; padding: 0.4rem 0.8rem; width: fit-content; }
  li { margin: 0 0 1rem; }
  .score { color: #333; font-variant-numeric: tabular-nums; }
</style>
</head>
<body>
<h1>SSV2</h1>
<form id="ssv2-search" novalidate>
  <label for="query">Query</label>
  <input id="query" name="query" autocomplete="off">
  <label for="k">Results (1-100)</label>
  <input id="k" name="k" type="number" min="1" max="100" value="5">
  <button type="submit">Search</button>
</form>
<p id="status" role="status"></p>
<ol id="hits"></ol>
<script>
const form = document.getElementById("ssv2-search");
const status = document.getElementById("status");
const hits = document.getElementById("hits");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  hits.replaceChildren();
  const query = document.getElementById("query").value.trim();
  const k = Number(document.getElementById("k").value);
  if (!query) {
    status.textContent = "Enter a query.";
    return;
  }
  if (!Number.isInteger(k) || k < 1 || k > 100) {
    status.textContent = "Results must be a whole number from 1 to 100.";
    return;
  }
  status.textContent = "Searching...";
  try {
    const response = await fetch("/v1/search", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query, k }),
    });
    const body = await response.json();
    if (!response.ok) {
      status.textContent = "Search failed (" + response.status + ").";
      return;
    }
    const timing = response.headers.get("x-request-duration-ms");
    if (body.hits.length === 0) {
      status.textContent = "No results.";
      return;
    }
    const suffix = timing ? " in " + timing + " ms." : ".";
    status.textContent = body.hits.length + " result(s)" + suffix;
    for (const hit of body.hits) {
      const item = document.createElement("li");
      const title = document.createElement("strong");
      title.textContent = hit.title || hit.id;
      const score = document.createElement("div");
      score.className = "score";
      score.textContent = "score " + Number(hit.score).toFixed(4);
      const text = document.createElement("p");
      text.textContent = hit.text;
      item.append(title, score, text);
      hits.append(item);
    }
  } catch {
    status.textContent = "Could not reach the search service.";
  }
});
</script>
</body>
</html>
"""

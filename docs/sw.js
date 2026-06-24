// coachvision service worker: cache the app shell so it installs and opens
// offline. GitHub API calls are always network (never cached) so data is live.
const CACHE = "coachvision-v1";
const SHELL = [
  "./",
  "./index.html",
  "./app.js",
  "./styles.css",
  "./manifest.webmanifest",
  "./icon.svg",
];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);

  // Never cache GitHub API/content responses — always fetch live data.
  if (url.hostname.endsWith("github.com") || url.hostname.endsWith("githubusercontent.com")) {
    return; // default network handling
  }

  // App shell: cache-first, fall back to network and update the cache.
  e.respondWith(
    caches.match(req).then((hit) =>
      hit ||
      fetch(req).then((res) => {
        const copy = res.clone();
        caches.open(CACHE).then((c) => c.put(req, copy)).catch(() => {});
        return res;
      }).catch(() => caches.match("./index.html"))
    )
  );
});

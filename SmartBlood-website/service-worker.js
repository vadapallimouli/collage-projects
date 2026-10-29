const CACHE_NAME = "e-smartblood-v4";

const APP_SHELL = [
    "/",
    "/index.html",
    "/styles.css",
    "/app.js",
    "/manifest.json"
];

self.addEventListener("install", event => {

    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => {
                return cache.addAll(APP_SHELL);
            })
    );

    self.skipWaiting();
});


self.addEventListener("activate", event => {

    event.waitUntil(

        caches.keys().then(cacheNames => {

            return Promise.all(

                cacheNames
                    .filter(name => name !== CACHE_NAME)
                    .map(name => caches.delete(name))

            );

        })

    );

    self.clients.claim();
});


self.addEventListener("fetch", event => {

    if (event.request.method !== "GET") {
        return;
    }

    event.respondWith(

        caches.match(event.request)
            .then(cachedResponse => {

                if (cachedResponse) {
                    return cachedResponse;
                }

                return fetch(event.request)
                    .then(networkResponse => {

                        return networkResponse;

                    })
                    .catch(() => {

                        return caches.match("/index.html");

                    });

            })

    );
});
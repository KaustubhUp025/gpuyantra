# The gpuyantra results explorer, as a static site for Cloud Run (Task 13b).
#
# Two files and nginx. `index.html` fetches `kernelsmith_explorer.jsx` from the same
# origin and compiles the JSX in the browser, so there is no build step here and the JSX
# in the repo remains the only copy of every number it displays.
#
#   docker build -t gpuyantra-explorer web/
#   docker run --rm -p 8081:8080 gpuyantra-explorer     # http://localhost:8081
#
# or `make deploy-explorer` for Cloud Run.
FROM nginx:1.29-alpine

# Cloud Run sends traffic to $PORT, which is 8080 unless told otherwise; nginx:alpine
# listens on 80. Rewritten rather than replaced with a whole conf file, so the image
# keeps upstream's defaults (gzip, mime types, index handling) as they are.
RUN sed -i 's/listen\s*80;/listen 8080;/; s/listen\s*\[::\]:80;/listen [::]:8080;/' \
      /etc/nginx/conf.d/default.conf

# The JSX must be served with a MIME type the browser will hand to `fetch` without
# complaint. nginx has no mapping for .jsx and would send application/octet-stream;
# text/plain is what this needs to be, since it is fetched as text and compiled by Babel.
RUN printf 'types { text/plain jsx; }\n' > /etc/nginx/conf.d/jsx-mime.conf \
 && sed -i '/^http {/a \    include /etc/nginx/conf.d/jsx-mime.conf;' /etc/nginx/nginx.conf

COPY index.html /usr/share/nginx/html/
COPY kernelsmith_explorer.jsx /usr/share/nginx/html/

EXPOSE 8080
CMD ["nginx", "-g", "daemon off;"]

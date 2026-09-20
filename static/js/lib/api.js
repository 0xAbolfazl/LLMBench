// Fetch helpers speaking the API's shared envelope:
//   {ok, error, data, ...extra}

/**
 * Fetch a JSON endpoint and unwrap its envelope.
 *
 * @param {string} path Endpoint path.
 * @param {RequestInit} [options] Fetch options.
 * @returns {Promise<object>} The full envelope on success.
 * @throws {Error} With the server's message when ok is false.
 */
export async function api(path, options = {}) {
  const res = await fetch(path, options);
  const data = await res.json();

  if (!data.ok) {
    throw new Error(data.error || `Request failed (${res.status})`);
  }

  return data;
}

/**
 * POST a JSON body to an endpoint.
 *
 * @param {string} path Endpoint path.
 * @param {object} payload Body to serialise.
 * @returns {Promise<object>} The unwrapped envelope.
 */
export function postJson(path, payload) {
  return api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

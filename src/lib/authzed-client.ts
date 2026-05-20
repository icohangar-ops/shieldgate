import { v1 } from "@authzed/authzed-node";

let client: ReturnType<typeof v1.NewClient> | null = null;

export function getAuthZedClient() {
  if (client) return client;

  const token = process.env.AUTHZED_API_KEY;
  const endpoint = process.env.AUTHZED_ENDPOINT || "grpc.authzed.com:443";

  if (!token) {
    throw new Error(
      "AUTHZED_API_KEY is not set. Set it in .env.local or use simulation mode."
    );
  }

  client = v1.NewClient(token, endpoint);
  return client;
}

export function isAuthZedConfigured(): boolean {
  return !!process.env.AUTHZED_API_KEY;
}

export { v1 };

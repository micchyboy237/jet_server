import { defineConfig } from "@playwright/test";

export default defineConfig({
  use: {
    browserName: "chromium",
    headless: false,
  },
  timeout: 30000,
});

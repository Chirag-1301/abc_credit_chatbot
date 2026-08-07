const { defineConfig } = require("cypress");

module.exports = defineConfig({
  e2e: {
    baseUrl: "http://localhost:8000",
    viewportWidth: 1280,
    viewportHeight: 800,
    video: false,
    supportFile: false,
    screenshotOnRunFailure: true,
    setupNodeEvents(on, config) {
      // node event listeners
    },
  },
});

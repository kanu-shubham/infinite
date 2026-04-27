"use strict";

const config = require("./config");
const { PricingService } = require("./pricingService");
const { createServer } = require("./routes");

function main() {
  const service = new PricingService(config);
  service.bootstrap();

  // Seed a default A/B experiment so /experiments has something to show.
  service.repo.setExperiment("pricing_strategy_v1", {
    variants: [
      { name: "control_static", weight: 1 },
      { name: "treatment_dynamic", weight: 1 },
    ],
    status: "active",
  });

  const server = createServer(service);
  server.listen(config.port, () => {
    const stats = {
      hotels: service.repo.listHotels().length,
      historicalLogs: service.repo.feedback.length,
      elasticityFits: service.elasticityFits.size,
    };
    console.log(`[pricing] listening on http://localhost:${config.port}`);
    console.log(`[pricing] bootstrap`, stats);
  });
}

if (require.main === module) main();

module.exports = { main };

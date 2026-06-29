import {
  parse
} from "./chunk-2SPSUN4U.js";
import "./chunk-VFX2UPZ3.js";
import "./chunk-ZH7SVVH7.js";
import "./chunk-3SX2RRW4.js";
import "./chunk-JF7M3FU3.js";
import "./chunk-2VBS5NCY.js";
import "./chunk-ROFPAOJC.js";
import "./chunk-UTS3S3JO.js";
import "./chunk-FWXVORCY.js";
import "./chunk-S5WWHIRC.js";
import "./chunk-IG4SYBEJ.js";
import "./chunk-5SSAL7U3.js";
import "./chunk-XGDJSZVJ.js";
import "./chunk-DQU7WUGC.js";
import "./chunk-RAMB7O63.js";
import "./chunk-KYEZTJNO.js";
import "./chunk-3L74IPNC.js";
import {
  selectSvgElement
} from "./chunk-AZSUYZZX.js";
import {
  configureSvgSize
} from "./chunk-CGQCCDEI.js";
import {
  log
} from "./chunk-GVR2T37F.js";
import {
  __name
} from "./chunk-UEB47VSI.js";
import "./chunk-WOOG5QLI.js";

// node_modules/mermaid/dist/chunks/mermaid.core/infoDiagram-FWYZ7A6U.mjs
var parser = {
  parse: __name(async (input) => {
    const ast = await parse("info", input);
    log.debug(ast);
  }, "parse")
};
var DEFAULT_INFO_DB = {
  version: "11.16.0" + (true ? "" : "-tiny")
};
var getVersion = __name(() => DEFAULT_INFO_DB.version, "getVersion");
var db = {
  getVersion
};
var draw = __name((text, id, version) => {
  log.debug("rendering info diagram\n" + text);
  const svg = selectSvgElement(id);
  configureSvgSize(svg, 100, 400, true);
  const group = svg.append("g");
  group.append("text").attr("x", 100).attr("y", 40).attr("class", "version").attr("font-size", 32).style("text-anchor", "middle").text(`v${version}`);
}, "draw");
var renderer = { draw };
var diagram = {
  parser,
  db,
  renderer
};
export {
  diagram
};
//# sourceMappingURL=infoDiagram-FWYZ7A6U-43FP3IPP.js.map

#!/usr/bin/env python3
"""
Generate induction loop detectors every 50m along the main_to_end route.

Parses network.net.xml to get edge lengths and lane counts, then writes
E1 induction loops into the existing detectors.add.xml file. The detector
file is modified in place.
"""
import xml.etree.ElementTree as ET
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
NETWORK_PATH = os.path.join(
    REPO_ROOT, "1_data_source", "multi_ramp",
    "sumo_network", "data", "network.net.xml",
)
DETECTOR_PATH = os.path.join(
    REPO_ROOT, "1_data_source", "multi_ramp",
    "sumo_network", "additional", "detectors.add.xml",
)

ROUTE_EDGES = [
    "edge_mainline_1", "edge_merge_1", "edge_mainline_2", "edge_out_1",
    "edge_mainline_3", "edge_out_2", "edge_mainline_4", "edge_merge_2",
    "edge_mainline_5", "edge_merge_3", "edge_mainline_6", "edge_out_3",
    "edge_mainline_7", "edge_merge_4", "edge_mainline_8",
]

INTERVAL = 50  # metres between detectors


def parse_network(net_path):
    """Return {edge_id: {"length": float, "num_lanes": int}} for every
    non-internal edge in the network."""
    tree = ET.parse(net_path)
    edges = {}
    for edge_el in tree.getroot().findall("edge"):
        eid = edge_el.get("id")
        if eid.startswith(":"):
            continue
        lanes = edge_el.findall("lane")
        if lanes:
            length = float(lanes[0].get("length"))
            edges[eid] = {"length": length, "num_lanes": len(lanes)}
    return edges


def generate_detectors(edges_info):
    """Walk along the route at INTERVAL-m steps and place one E1 detector
    per lane at each position."""
    detectors = []
    cumulative = 0.0

    for edge_id in ROUTE_EDGES:
        info = edges_info[edge_id]
        edge_length = info["length"]
        num_lanes = info["num_lanes"]
        edge_end = cumulative + edge_length

        first_mark = int(cumulative / INTERVAL) * INTERVAL
        if first_mark < cumulative:
            first_mark += INTERVAL

        dist = first_mark
        while dist < edge_end:
            pos_on_edge = dist - cumulative
            # SUMO needs pos strictly inside (0, length)
            if pos_on_edge < 0.1:
                pos_on_edge = 0.1
            if pos_on_edge > edge_length - 0.1:
                dist += INTERVAL
                continue

            for lane in range(num_lanes):
                det_id = f"det_dist_{dist}_L{lane}"
                lane_id = f"{edge_id}_{lane}"
                detectors.append(
                    f'    <inductionLoop id="{det_id}" '
                    f'lane="{lane_id}" '
                    f'pos="{pos_on_edge:.2f}" freq="15" file="NUL"/>'
                )
            dist += INTERVAL

        cumulative = edge_end

    return detectors, cumulative


def strip_existing_spatial_detectors(content):
    """Remove any previously generated spatial detector block."""
    marker = "<!-- Spatial speed tracking detectors"
    if marker not in content:
        return content
    start = content.index(marker)
    # Find the next blank line after the block (before </additional>)
    closing = content.find("</additional>", start)
    return content[:start] + content[closing:]


def main():
    edges_info = parse_network(NETWORK_PATH)
    detectors, total_length = generate_detectors(edges_info)

    with open(DETECTOR_PATH, "r") as f:
        content = f.read()

    content = strip_existing_spatial_detectors(content)

    closing_tag = "</additional>"
    insert_pos = content.rfind(closing_tag)
    new_content = (
        content[:insert_pos]
        + "    <!-- Spatial speed tracking detectors (every 50m along main_to_end) -->\n"
        + "\n".join(detectors)
        + "\n\n"
        + closing_tag
    )

    with open(DETECTOR_PATH, "w") as f:
        f.write(new_content)

    unique_positions = set()
    for d in detectors:
        det_id = d.split('"')[1]
        base = det_id.rsplit("_L", 1)[0]
        unique_positions.add(base)
    print(f"Generated {len(detectors)} induction loops "
          f"at {len(unique_positions)} positions along {total_length:.1f}m route")


if __name__ == "__main__":
    main()

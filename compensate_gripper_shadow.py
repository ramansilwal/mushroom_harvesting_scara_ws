import numpy as np

from rrt_utils import forward_kinematics


def compensate_gripper_shadow(obstacles, old_obstacles, angles, lengths):

    obstacles = np.asarray(obstacles).reshape(-1, 2, 2)

    if old_obstacles is None:
        return obstacles

    old_obstacles = np.asarray(old_obstacles).reshape(-1, 2, 2)

    # 1. Laser position and heading
    fk = forward_kinematics(angles, lengths)
    laser_pos = fk[-2]
    heading = np.sum(angles)

    # 2. Midpoints of previous obstacles
    points = old_obstacles.mean(axis=1)

    # 3. Vectors from laser to previous obstacles
    vectors = points - laser_pos

    # 4. Angles relative to laser
    theta = np.arctan2(vectors[:, 1], vectors[:, 0])

    # 5. Angular difference from current heading
    diff = np.arctan2(
        np.sin(theta - heading),
        np.cos(theta - heading)
    )

    # 6. Select previous obstacles within ±22 degrees
    mask = np.abs(diff) <= np.deg2rad(22)

    selected = old_obstacles[mask]

    # 7. Append selected previous obstacles to current obstacles
    result = np.concatenate([obstacles, selected], axis=0)

    # -----------------------------
    # Visualization
    # -----------------------------

    # fig, axs = plt.subplots(2, 2, figsize=(14, 12))

    # def plot_segments(ax, segments, color, label, linewidth=1.5):
    #     if len(segments):
    #         ax.add_collection(
    #             LineCollection(
    #                 segments,
    #                 colors=color,
    #                 linewidths=linewidth,
    #                 label=label
    #             )
    #         )

    # def plot_robot(ax):
    #     ax.plot(
    #         fk[:, 0], fk[:, 1],
    #         "k-o",
    #         linewidth=2,
    #         markersize=4,
    #         label="Robot"
    #     )

    #     ax.scatter(
    #         *laser_pos,
    #         color="orange",
    #         s=80,
    #         zorder=5,
    #         label="Laser"
    #     )

    # def plot_shadow(ax):
    #     radius = 1.8

    #     theta_range = np.linspace(
    #         heading - np.deg2rad(22),
    #         heading + np.deg2rad(22),
    #         100
    #     )

    #     x = laser_pos[0] + radius * np.cos(theta_range)
    #     y = laser_pos[1] + radius * np.sin(theta_range)

    #     ax.fill(
    #         np.r_[laser_pos[0], x, laser_pos[0]],
    #         np.r_[laser_pos[1], y, laser_pos[1]],
    #         color="orange",
    #         alpha=0.2,
    #         label="±22° shadow"
    #     )

    #     # Current heading
    #     ax.plot(
    #         [
    #             laser_pos[0],
    #             laser_pos[0] + radius * np.cos(heading)
    #         ],
    #         [
    #             laser_pos[1],
    #             laser_pos[1] + radius * np.sin(heading)
    #         ],
    #         "m--",
    #         linewidth=2,
    #         label="Current heading"
    #     )

    # # Panel 1: Current and previous scans
    # ax = axs[0, 0]
    # plot_segments(ax, old_obstacles, "blue", "Previous")
    # plot_segments(ax, obstacles, "red", "Current")
    # plot_robot(ax)
    # ax.set_title("1. Current vs Previous Obstacles")

    # # Panel 2: Shadow region
    # ax = axs[0, 1]
    # plot_segments(ax, old_obstacles, "blue", "Previous")
    # plot_robot(ax)
    # plot_shadow(ax)
    # ax.set_title("2. Current Heading and ±22° Region")

    # # Panel 3: Mask selection
    # ax = axs[1, 0]
    # plot_segments(ax, old_obstacles[~mask], "lightgray", "Not selected")
    # plot_segments(ax, selected, "green", "Selected previous", 2)
    # plot_robot(ax)
    # plot_shadow(ax)
    # ax.set_title(f"3. Selected {mask.sum()} / {len(mask)} Previous Segments")

    # # Panel 4: Final result
    # ax = axs[1, 1]
    # plot_segments(ax, obstacles, "red", "Current")
    # plot_segments(ax, selected, "green", "Recovered", 2)
    # plot_robot(ax)
    # ax.set_title(f"4. Final Map: {len(obstacles)} + {len(selected)} Segments")

    # # Consistent limits for all panels
    # all_points = np.concatenate([
    #     obstacles.reshape(-1, 2),
    #     old_obstacles.reshape(-1, 2),
    #     fk,
    #     laser_pos.reshape(1, 2)
    # ])

    # xmin, ymin = all_points.min(axis=0) - 0.15
    # xmax, ymax = all_points.max(axis=0) + 0.15

    # for ax in axs.flat:
    #     ax.set_xlim(xmin, xmax)
    #     ax.set_ylim(ymin, ymax)
    #     ax.set_aspect("equal")
    #     ax.grid(alpha=0.3)
    #     ax.legend(loc="best")

    # plt.tight_layout()
    # plt.show()

    # print(f"Laser position: {laser_pos}")
    # print(f"Heading: {np.rad2deg(heading):.2f}°")
    # print(f"Shadow: ±22°")
    # print(f"Previous obstacles: {len(old_obstacles)}")
    # print(f"Selected previous obstacles: {mask.sum()}")
    # print(f"Current obstacles: {len(obstacles)}")
    # print(f"Final obstacles: {len(result)}")

    return result


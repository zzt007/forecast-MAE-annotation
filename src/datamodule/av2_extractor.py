import traceback
from pathlib import Path
from typing import List

import av2.geometry.interpolate as interp_utils
import numpy as np
import torch
from av2.map.map_api import ArgoverseStaticMap

from .av2_data_utils import (
    OBJECT_TYPE_MAP,
    OBJECT_TYPE_MAP_COMBINED,
    LaneTypeMap,
    load_av2_df,
)


class Av2Extractor:
    def __init__(
        self,
        radius: float = 150,
        save_path: Path = None,
        mode: str = "train",
        ignore_type: List[int] = [5, 6, 7, 8, 9], # AV2数据中有10类标签（数字是因为在av2_data_utils.py中做了类型和数字的映射），这里忽略后5类，包括Static、Background、Construction、Riderless_bicycle、Unknown；只留了前5类，vehicle、pedestrian、motorcycle、cyclist、bus
        remove_outlier_actors: bool = True,
    ) -> None:
        self.save_path = save_path
        self.mode = mode
        self.radius = radius
        self.remove_outlier_actors = remove_outlier_actors
        self.ignore_type = ignore_type

    def save(self, file: Path):
        assert self.save_path is not None

        try:
            data = self.get_data(file)
        except Exception:
            print(traceback.format_exc())
            print("found error while extracting data from {}".format(file))
        save_file = self.save_path / (file.stem + ".pt") # file.stem 的作用：提取文件路径中文件名的主干部分（即去掉扩展名的部分）
        torch.save(data, save_file)

    def get_data(self, file: Path):
        return self.process(file)

    def process(self, raw_path: str, agent_id=None):
        df, am, scenario_id = load_av2_df(raw_path) # 返回dataframe和ArgoverseStaticMap对象，以及场景ID
        city = df.city.values[0]
        agent_id = df["focal_track_id"].values[0] # 专注于focal agent

        local_df = df[df["track_id"] == agent_id].iloc # 找到focal agent的所有信息行

        # 找到focal agent的last observe point
        origin = torch.tensor(
            [local_df[49]["position_x"], local_df[49]["position_y"]], dtype=torch.float
        )
        theta = torch.tensor([local_df[49]["heading"]], dtype=torch.float) # AV2还提供了object bounding box的航向角，单位是弧度，是在地图坐标系下map coordinate frame
        # 计算旋转矩阵
        rotate_mat = torch.tensor(
            [
                [torch.cos(theta), -torch.sin(theta)],
                [torch.sin(theta), torch.cos(theta)],
            ],
        )

        timestamps = list(np.sort(df["timestep"].unique()))
        cur_df = df[df["timestep"] == timestamps[49]]
        actor_ids = list(cur_df["track_id"].unique()) # 找到当前时刻下其它actor的track id
        cur_pos = torch.from_numpy(cur_df[["position_x", "position_y"]].values).float()
        out_of_range = np.linalg.norm(cur_pos - origin, axis=1) > self.radius # 筛掉在focal agent半径阈值以外的agent
        actor_ids = [aid for i, aid in enumerate(actor_ids) if not out_of_range[i]]
        actor_ids.remove(agent_id) # 去掉focal agent
        actor_ids = [agent_id] + actor_ids # 将focal agent 重新添加到列表的开头，确保其始终是第一个元素，尽管不明确放第一有什么用
        num_nodes = len(actor_ids) # 当前场景下满足条件的actor数量

        df = df[df["track_id"].isin(actor_ids)] # 得到在actor_ids中的actor的信息

        # initialization
        x = torch.zeros(num_nodes, 110, 2, dtype=torch.float)
        x_attr = torch.zeros(num_nodes, 3, dtype=torch.uint8)
        x_heading = torch.zeros(num_nodes, 110, dtype=torch.float)
        x_velocity = torch.zeros(num_nodes, 110, dtype=torch.float)
        x_track_horizon = torch.zeros(num_nodes, dtype=torch.int)
        padding_mask = torch.ones(num_nodes, 110, dtype=torch.bool)

        for actor_id, actor_df in df.groupby("track_id"):
            node_idx = actor_ids.index(actor_id) # 从 actor_ids 列表中查找 actor_id 的索引，并将其赋值给 node_idx。如果 actor_id 不在 actor_ids 中，会抛出 ValueError 异常。
            node_steps = [timestamps.index(ts) for ts in actor_df["timestep"]]
            object_type = OBJECT_TYPE_MAP[actor_df["object_type"].values[0]]
            x_attr[node_idx, 0] = object_type # 轨迹序列对应的object的类型，如vehicle等
            x_attr[node_idx, 1] = actor_df["object_category"].values[0] #轨迹序列分配种类，用于给轨迹预测的数据质量提供参考，一般来说，有四种：SCORED_TRACK，UNSCORED_TRACK，FOCAL_TRACK，TRACK_FRAGMENT
            x_attr[node_idx, 2] = OBJECT_TYPE_MAP_COMBINED[
                actor_df["object_type"].values[0]
            ] # 进一步将类别映射，参见av2_data_utils.py
            
            x_track_horizon[node_idx] = node_steps[-1] - node_steps[0] # 轨迹序列的长度

            padding_mask[node_idx, node_steps] = False
            if padding_mask[node_idx, 49] or object_type in self.ignore_type:
                padding_mask[node_idx, 50:] = True

            pos_xy = torch.from_numpy(
                np.stack(
                    [actor_df["position_x"].values, actor_df["position_y"].values],
                    axis=-1,
                )
            ).float()
            heading = torch.from_numpy(actor_df["heading"].values).float()
            velocity = torch.from_numpy(
                actor_df[["velocity_x", "velocity_y"]].values
            ).float()
            velocity_norm = torch.norm(velocity, dim=1) # 速度归一化

            x[node_idx, node_steps, :2] = torch.matmul(pos_xy - origin, rotate_mat) # 坐标变换
            x_heading[node_idx, node_steps] = (heading - theta + np.pi) % (
                2 * np.pi
            ) - np.pi # 变换到相对focal agent的角度
            x_velocity[node_idx, node_steps] = velocity_norm

        (
            lane_positions,
            is_intersections,
            lane_ctrs,
            lane_angles,
            lane_attr,
            lane_padding_mask,
        ) = self.get_lane_features(am, origin, origin, rotate_mat, self.radius)

        # 筛掉离车道线过远（这里定义为5m）的智能体，但一定要保留focal agent
        if self.remove_outlier_actors:
            lane_samples = lane_positions[:, ::1, :2].view(-1, 2) 
            nearest_dist = torch.cdist(x[:, 49, :2], lane_samples).min(dim=1).values # cdist计算欧式距离
            valid_actor_mask = nearest_dist < 5
            valid_actor_mask[0] = True  # always keep the target agent

            x = x[valid_actor_mask]
            x_heading = x_heading[valid_actor_mask]
            x_velocity = x_velocity[valid_actor_mask]
            x_attr = x_attr[valid_actor_mask]
            padding_mask = padding_mask[valid_actor_mask]
            num_nodes = x.shape[0]

        x_ctrs = x[:, 49, :2].clone()
        x_positions = x[:, :50, :2].clone()
        x_velocity_diff = x_velocity[:, :50].clone()

        x[:, 50:] = torch.where(
            (padding_mask[:, 49].unsqueeze(-1) | padding_mask[:, 50:]).unsqueeze(-1),
            torch.zeros(num_nodes, 60, 2),
            x[:, 50:] - x[:, 49].unsqueeze(-2),
        )
        x[:, 1:50] = torch.where(
            (padding_mask[:, :49] | padding_mask[:, 1:50]).unsqueeze(-1),
            torch.zeros(num_nodes, 49, 2),
            x[:, 1:50] - x[:, :49],
        )
        x[:, 0] = torch.zeros(num_nodes, 2)

        x_velocity_diff[:, 1:50] = torch.where(
            (padding_mask[:, :49] | padding_mask[:, 1:50]),
            torch.zeros(num_nodes, 49),
            x_velocity_diff[:, 1:50] - x_velocity_diff[:, :49],
        )
        x_velocity_diff[:, 0] = torch.zeros(num_nodes)

        y = None if self.mode == "test" else x[:, 50:] # 如果在test，则y为None；否则为未来轨迹

        return {
            "x": x[:, :50],
            "y": y,
            "x_attr": x_attr, # [node_num, 3]，其中3种attr分别是 object_type \ object_category \ object_type_map_combined
            "x_positions": x_positions,
            "x_centers": x_ctrs,
            "x_angles": x_heading,
            "x_velocity": x_velocity,
            "x_velocity_diff": x_velocity_diff,
            "x_padding_mask": padding_mask,
            "lane_positions": lane_positions,
            "lane_centers": lane_ctrs,
            "lane_angles": lane_angles,
            "lane_attr": lane_attr,
            "lane_padding_mask": lane_padding_mask,
            "is_intersections": is_intersections,
            "origin": origin.view(-1, 2),
            "theta": theta,
            "scenario_id": scenario_id,
            "track_id": agent_id,
            "city": city,
        }

    @staticmethod
    def get_lane_features(
        am: ArgoverseStaticMap,
        query_pos: torch.Tensor,
        origin: torch.Tensor,
        rotate_mat: torch.Tensor,
        radius: float,
    ):
        # 通过map_api，以某个query为中心搜索附近一定半径内的lane_segments
        lane_segments = am.get_nearby_lane_segments(query_pos.numpy(), radius)

        lane_positions, is_intersections, lane_attrs = [], [], []
        for segment in lane_segments:
            lane_centerline, lane_width = interp_utils.compute_midpoint_line(
                left_ln_boundary=segment.left_lane_boundary.xyz,
                right_ln_boundary=segment.right_lane_boundary.xyz,
                num_interp_pts=20,
            )
            lane_centerline = torch.from_numpy(lane_centerline[:, :2]).float()
            lane_centerline = torch.matmul(lane_centerline - origin, rotate_mat) # 坐标变换到focal agent-centric
            is_intersection = am.lane_is_in_intersection(segment.id)

            lane_positions.append(lane_centerline)
            is_intersections.append(is_intersection)

            # get lane attrs
            lane_type = LaneTypeMap[segment.lane_type] # 映射关系参见av2_data_utils.py, 具体为： LaneType.VEHICLE.value: 0,LaneType.BIKE.value: 1,LaneType.BUS.value: 2,
            attribute = torch.tensor(
                [lane_type, lane_width, is_intersection], dtype=torch.float
            )
            lane_attrs.append(attribute)

        lane_positions = torch.stack(lane_positions)
        lanes_ctr = lane_positions[:, 9:11].mean(dim=1) # 计算lanes的中心点，因为插值了20个点
        lanes_angle = torch.atan2(
            lane_positions[:, 10, 1] - lane_positions[:, 9, 1],
            lane_positions[:, 10, 0] - lane_positions[:, 9, 0],
        )
        is_intersections = torch.Tensor(is_intersections)
        lane_attrs = torch.stack(lane_attrs, dim=0)

        x_max, x_min = radius, -radius
        y_max, y_min = radius, -radius

        padding_mask = (
            (lane_positions[:, :, 0] > x_max)
            | (lane_positions[:, :, 0] < x_min)
            | (lane_positions[:, :, 1] > y_max)
            | (lane_positions[:, :, 1] < y_min)
        )

        invalid_mask = padding_mask.all(dim=-1)
        lane_positions = lane_positions[~invalid_mask]
        is_intersections = is_intersections[~invalid_mask]
        lane_attrs = lane_attrs[~invalid_mask]
        lanes_ctr = lanes_ctr[~invalid_mask]
        lanes_angle = lanes_angle[~invalid_mask]
        padding_mask = padding_mask[~invalid_mask]

        lane_positions = torch.where(
            padding_mask[..., None], torch.zeros_like(lane_positions), lane_positions
        )

        return (
            lane_positions,
            is_intersections,
            lanes_ctr,
            lanes_angle,
            lane_attrs,
            padding_mask,
        )

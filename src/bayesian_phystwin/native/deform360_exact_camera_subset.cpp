#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

namespace {

class Solver {
 public:
  int camera_count = 0;
  int center_count = 0;
  int selected_count = 0;
  std::vector<std::uint64_t> masks;
  std::vector<int> support_counts;
  std::vector<double> pair_angles;
  std::vector<int> selected;
  std::vector<int> best_selected;
  std::vector<std::vector<int>> suffix_support;
  std::vector<std::vector<int>> suffix_best_sum;
  int best_first = -1;
  int best_second = -1;
  int best_third = -1;
  double best_fourth = -std::numeric_limits<double>::infinity();

  double angle(int center, int first, int second) const {
    const std::size_t index =
        (static_cast<std::size_t>(center) * camera_count + first) *
            camera_count +
        second;
    return pair_angles[index];
  }

  bool branch_can_improve(int start, int depth, std::uint64_t once,
                          std::uint64_t twice, std::uint64_t thrice,
                          int support_sum) const {
    const int remaining = selected_count - depth;
    if (support_sum + suffix_best_sum[start][remaining] < best_third &&
        best_first == center_count && best_second == center_count) {
      return false;
    }
    if (depth > 3 || best_first < 0) {
      return true;
    }
    int upper_first = 0;
    int upper_second = 0;
    for (int center = 0; center < center_count; ++center) {
      const std::uint64_t bit = std::uint64_t{1} << center;
      int current = 0;
      if (thrice & bit) {
        current = 3;
      } else if (twice & bit) {
        current = 2;
      } else if (once & bit) {
        current = 1;
      }
      const int attainable = current + suffix_support[start][center];
      upper_first += attainable >= 2;
      upper_second += attainable >= 3;
    }
    return upper_first > best_first ||
           (upper_first == best_first && upper_second >= best_second);
  }

  double fourth_score(std::uint64_t twice) const {
    std::vector<double> values;
    values.reserve(center_count);
    for (int center = 0; center < center_count; ++center) {
      const std::uint64_t bit = std::uint64_t{1} << center;
      if (!(twice & bit)) {
        continue;
      }
      double maximum = 0.0;
      for (int left = 0; left < selected_count; ++left) {
        const int first = selected[left];
        if (!(masks[first] & bit)) {
          continue;
        }
        for (int right = left + 1; right < selected_count; ++right) {
          const int second = selected[right];
          if (masks[second] & bit) {
            maximum = std::max(maximum, angle(center, first, second));
          }
        }
      }
      values.push_back(maximum);
    }
    if (values.empty()) {
      return 0.0;
    }
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 == 1) {
      return values[middle];
    }
    return (values[middle - 1] + values[middle]) / 2.0;
  }

  void evaluate(std::uint64_t twice, std::uint64_t thrice, int support_sum) {
    const int first = __builtin_popcountll(twice);
    const int second = __builtin_popcountll(thrice);
    if (first < best_first ||
        (first == best_first && second < best_second) ||
        (first == best_first && second == best_second &&
         support_sum < best_third)) {
      return;
    }
    const double fourth = fourth_score(twice);
    const bool better =
        first > best_first ||
        (first == best_first && second > best_second) ||
        (first == best_first && second == best_second &&
         support_sum > best_third) ||
        (first == best_first && second == best_second &&
         support_sum == best_third && fourth > best_fourth);
    if (better) {
      best_first = first;
      best_second = second;
      best_third = support_sum;
      best_fourth = fourth;
      best_selected = selected;
    }
  }

  void search(int start, int depth, std::uint64_t once, std::uint64_t twice,
              std::uint64_t thrice, int support_sum) {
    if (depth == selected_count) {
      evaluate(twice, thrice, support_sum);
      return;
    }
    if (!branch_can_improve(start, depth, once, twice, thrice, support_sum)) {
      return;
    }
    const int needed = selected_count - depth;
    for (int camera = start; camera <= camera_count - needed; ++camera) {
      const std::uint64_t mask = masks[camera];
      selected[depth] = camera;
      search(camera + 1, depth + 1, once | mask, twice | (once & mask),
             thrice | (twice & mask), support_sum + support_counts[camera]);
    }
  }

  void prepare_bounds() {
    suffix_support.assign(camera_count + 1,
                          std::vector<int>(center_count, 0));
    for (int camera = camera_count - 1; camera >= 0; --camera) {
      suffix_support[camera] = suffix_support[camera + 1];
      for (int center = 0; center < center_count; ++center) {
        suffix_support[camera][center] +=
            (masks[camera] >> center) & std::uint64_t{1};
      }
    }
    suffix_best_sum.assign(camera_count + 1,
                           std::vector<int>(selected_count + 1, -1000000));
    suffix_best_sum[camera_count][0] = 0;
    for (int camera = camera_count - 1; camera >= 0; --camera) {
      suffix_best_sum[camera][0] = 0;
      for (int count = 1; count <= selected_count; ++count) {
        suffix_best_sum[camera][count] = std::max(
            suffix_best_sum[camera + 1][count],
            support_counts[camera] + suffix_best_sum[camera + 1][count - 1]);
      }
    }
  }
};

}  // namespace

int main() {
  Solver solver;
  if (!(std::cin >> solver.camera_count >> solver.center_count >>
        solver.selected_count)) {
    return 2;
  }
  if (solver.camera_count < solver.selected_count || solver.selected_count < 2 ||
      solver.center_count < 1 || solver.center_count > 63) {
    return 3;
  }
  solver.masks.resize(solver.camera_count);
  solver.support_counts.resize(solver.camera_count);
  for (auto& mask : solver.masks) {
    std::cin >> mask;
  }
  for (auto& count : solver.support_counts) {
    std::cin >> count;
  }
  const std::size_t angle_count =
      static_cast<std::size_t>(solver.center_count) * solver.camera_count *
      solver.camera_count;
  solver.pair_angles.resize(angle_count);
  for (auto& value : solver.pair_angles) {
    std::cin >> value;
  }
  if (!std::cin) {
    return 4;
  }
  solver.selected.resize(solver.selected_count);
  solver.prepare_bounds();
  solver.search(0, 0, 0, 0, 0, 0);
  if (solver.best_selected.size() !=
      static_cast<std::size_t>(solver.selected_count)) {
    return 5;
  }
  for (int index = 0; index < solver.selected_count; ++index) {
    if (index) {
      std::cout << ' ';
    }
    std::cout << solver.best_selected[index];
  }
  std::cout << '\n'
            << solver.best_first << ' ' << solver.best_second << ' '
            << solver.best_third << ' ' << std::setprecision(17)
            << solver.best_fourth << '\n';
  return 0;
}

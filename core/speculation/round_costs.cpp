// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "core/speculation/round_costs.h"

#include <algorithm>
#include <cmath>

namespace flyx {

void RoundCosts::reset(const std::vector<double>& base_ms, const std::vector<double>* per_token, int at,
                       const std::vector<int>* narrower) {
  std::lock_guard<std::mutex> lock(mu_);
  lines_.assign(base_ms.size(), Line{});
  for (size_t i = 0; i < base_ms.size(); ++i) {
    Line& l = lines_[i];
    const double a = base_ms[i], b = per_token && i < per_token->size() ? 1000.0 * (*per_token)[i] : 0.0;
    l.a0 = std::isfinite(a) && a > 0 ? a : 0.0;  // 0: untimed, its first reading takes the load's place
    l.b0 = std::isfinite(b) && b > 0 ? b : 0.0;
    const int n = narrower && i < narrower->size() ? (*narrower)[i] : -1;
    l.narrower = n >= 0 && size_t(n) < base_ms.size() && n != int(i) ? n : -1;
  }
  // Widths ascend along a chain, so following narrower ends; a malformed table (a cycle) is cut here.
  for (size_t i = 0; i < lines_.size(); ++i) {
    size_t steps = 0;
    for (int j = lines_[i].narrower; j >= 0; j = lines_[size_t(j)].narrower)
      if (++steps > lines_.size()) {
        lines_[i].narrower = -1;
        break;
      }
  }
  // An untimed round not read yet: the cheapest the load timed, so that it is chosen and read (1 ms where none was
  // timed); never nothing, which a rate would divide by.
  untimed_ = 0;
  for (const Line& l : lines_)
    if (l.a0 > 0 && (untimed_ == 0 || l.a0 < untimed_)) untimed_ = l.a0;
  if (!(untimed_ > 0)) untimed_ = 1.0;
  at_ = at;
}

size_t RoundCosts::size() const {
  std::lock_guard<std::mutex> lock(mu_);
  return lines_.size();
}

double RoundCosts::x_of(int context) const { return std::max(0, context - at_) / 1000.0; }

// The line minimizing kPriorRounds (a - a0)^2 + kSlopeRounds (b - b0)^2 + the observations' kept squared errors, its
// slope no less than 0 (a round reads no fewer keys at a longer context) and its cost at `at` no less than kFloor
// times the load's: one line across a knee (the GPU's part of the K/V, auto's switch) would otherwise tilt down to
// nothing at short contexts, and the clip keep their rounds there from bringing it back.  At x.
double RoundCosts::fit(const Line& l, double x) {
  if (!(l.a0 > 0)) return 0.0;  // untimed and not read yet (priced() gives it a price)
  const double p = kPriorRounds, q = kSlopeRounds;
  const double m11 = p + l.s1, m12 = l.sx, m22 = q + l.sxx;
  const double r1 = p * l.a0 + l.sy, r2 = q * l.b0 + l.sxy;
  double b = (m11 * r2 - m12 * r1) / (m11 * m22 - m12 * m12);
  double a = (r1 - m12 * b) / m11;
  if (b < 0) b = 0, a = r1 / m11;
  if (const double least = kFloor * l.a0; a < least) a = least, b = std::max(0.0, (r2 - m12 * least) / m22);
  return a + b * x;
}

// A wider round of a drafter's chain verifies more drafts than the next narrower one: it costs no less, and at least
// what the load timed between the two more.  An adaptive round that stopped early teaches the narrower round only;
// without this a wider one kept its short context's cost and was chosen at a long context where neither pays.
double RoundCosts::priced(size_t round, double x) const {
  const Line& l = lines_[round];
  double c = l.a0 > 0 ? fit(l, x) : untimed_;
  if (l.narrower >= 0) {
    const Line& n = lines_[size_t(l.narrower)];
    c = std::max(c, priced(size_t(l.narrower), x) + std::max(0.0, l.a0 - n.a0));
  }
  return c;
}

double RoundCosts::cost(size_t round, int context) const {
  std::lock_guard<std::mutex> lock(mu_);
  return round < lines_.size() ? priced(round, x_of(context)) : 1.0;
}

void RoundCosts::observe(size_t round, int context, double ms) {
  std::lock_guard<std::mutex> lock(mu_);
  if (round >= lines_.size() || !(ms > 0) || !std::isfinite(ms)) return;
  Line& l = lines_[round];
  if (!(l.a0 > 0)) l.a0 = ms;
  const double x = x_of(context), said = priced(round, x), y = std::clamp(ms, said / kClip, said * kClip);
  l.s1 = kKeep * l.s1 + 1;
  l.sx = kKeep * l.sx + x;
  l.sxx = kKeep * l.sxx + x * x;
  l.sy = kKeep * l.sy + y;
  l.sxy = kKeep * l.sxy + x * y;
}

}  // namespace flyx

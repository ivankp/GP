#ifndef RANGE_HH
#define RANGE_HH

#include <iterator>
#include <type_traits>

template <typename Begin, typename End>
class range {
  Begin b;
  End e;
public:
  template <typename B, typename E>
  constexpr range(B&& b, E&& e)
  : b(std::forward<B>(b)), e(std::forward<E>(e)) { }

  template <typename T>
  constexpr range(T&& xs)
  : b(begin(std::forward<T>(xs))), e(end(std::forward<T>(xs))) { }

  constexpr Begin& begin() const noexcept { return b; }
  constexpr End  & end  () const noexcept { return e; }
  constexpr size_t size() const noexcept { return std::distance(b,e); }

  friend constexpr Begin& begin(const range& r) noexcept { return r.begin(); }
  friend constexpr End  & end  (const range& r) noexcept { return r.end  (); }
  friend constexpr size_t size (const range& r) noexcept { return r.size (); }
};

template <typename T>
using range_for_t = range<
  std::decay_t<decltype(begin(std::declval<T>()))>,
  std::decay_t<decltype(end  (std::declval<T>()))>
>;

#endif

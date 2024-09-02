#pragma once

#include <madrona/sync.hpp>

namespace Acrobat {

  struct EpisodeManager {
    madrona::AtomicU32 curEpisode;
  };

  struct WorldInit {
    EpisodeManager *episodeMgr;
  };

}

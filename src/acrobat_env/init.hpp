#pragma once

#include <madrona/sync.hpp>

namespace Acrobat {

  struct EpisodeManager {
    madrona::AtomicU32 curEpisode;
    uint32_t episodeLength; // current length of this episode
  };

  struct WorldInit {
    EpisodeManager *episodeMgr;
  };

}

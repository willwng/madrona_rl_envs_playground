#include "mgr.hpp"

#include <madrona/macros.hpp>
#include <madrona/py/bindings.hpp>

namespace Balance {

  NB_MODULE(madrona_balance_example_python, m) {
    madrona::py::setupMadronaSubmodule(m);
    nb::class_<Manager> (m, "BalanceBeamSimulator")
      .def("__init__", [](Manager *self,
			  madrona::py::PyExecMode exec_mode,
			  int64_t gpu_id,
			  int64_t num_worlds,
			  bool debug_compile) {
	new (self) Manager(Manager::Config {
	    .execMode = exec_mode,
	    .gpuID = (int)gpu_id,
	    .numWorlds = (uint32_t)num_worlds,
	    .debugCompile = debug_compile,
	  });
      }, nb::arg("exec_mode"), nb::arg("gpu_id"), nb::arg("num_worlds"),
           nb::arg("debug_compile") = true)
      .def("step", &Manager::step)
      .def("done_tensor", &Manager::doneTensor)
      .def("active_agent_tensor", &Manager::activeAgentTensor)
      .def("action_tensor", &Manager::actionTensor)
      .def("observation_tensor", &Manager::observationTensor)
      .def("agent_state_tensor", &Manager::observationTensor)
      .def("action_mask_tensor", &Manager::actionMaskTensor)
      .def("reward_tensor", &Manager::rewardTensor)
      .def("world_id_tensor", &Manager::worldIDTensor)
      .def("agent_id_tensor", &Manager::agentIDTensor)
      ;
  }

}

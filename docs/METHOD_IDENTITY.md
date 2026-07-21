# STAGE method identity

The official method name is:

**Structure-Preserving Timestamp Alignment for Geometry Enhancement (STAGE)**

The three parts of the name map to the method as follows:

- **Structure-Preserving**: learns and retains less frequent normal dynamics
  instead of allowing patch abundance to suppress them.
- **Timestamp Alignment**: aligns only timestamps shared by shifted windows,
  reducing representation perturbations introduced by surrounding window
  context.
- **Geometry Enhancement**: regulates sampling through an adaptive intermediate
  partition and reconstructs the final embedding geometry with an adaptive
  exemplar-memory pool.

STAGE requires neither auxiliary negative samples nor a pre-specified prototype
count. The repository, result records, paper tables, figures, and deployment
scripts must use `STAGE` as the method identifier.

# Message containing labelspace information for a given model

string[] names    # List of all label names (empty string if no name for specific label value)
int32[] invalid   # All label IDs that should be ignored
string metadata   # Additional labelspace metadata serialized via json

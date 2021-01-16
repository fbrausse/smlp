syntax = "proto2";

message Request {
	enum Type {
		PING          = 0;
		SMTLIB_SCRIPT = 1;
		CLIENT_QUIT   = 2;
	}
	required Type   type  = 1;
	optional bytes  stdin = 2;
}

message CmdResult {
	required int32 status = 1;
	required bytes stdout = 2;
	required bytes stderr = 3;
}

message Reply {
	enum Type {
		PONG         = 0;
		SMTLIB_REPLY = 1;
		ERROR        = 2;
	}
	enum Code {
		IDLE        = 0;
		BUSY        = 1;
		UNKNOWN_REQ = 2;
	}
	required Type      type      = 1;
	optional Code      code      = 2;
	optional string    error_msg = 3;
	optional CmdResult cmd       = 4;
}

message Smlp {
	required int32 version  = 1;

	/* fresh for requests
	 * requests's msg_id for replies */
	required int32 msg_id   = 2;

	/* setting none or both is invalid */
	optional Request request = 4;
	optional Reply   reply   = 5;
}

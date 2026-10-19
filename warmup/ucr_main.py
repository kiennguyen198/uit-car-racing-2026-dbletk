"""UIT CAR RACING — competition entry point."""
from competition_console import begin,finish,failure

def run():
    begin()
    code=0
    try:
        from controller import main
        main()
    except KeyboardInterrupt:
        pass
    except Exception as error:
        import traceback
        traceback.print_exc()
        failure(error)
        code=1
    finally:
        finish()
    return code

if __name__=='__main__':
    raise SystemExit(run())
